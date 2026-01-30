import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

interface ConversationMessage {
    role: "user" | "assistant";
    content: string;
}

interface PipelineResult {
    document_type: string;
    total_pages: number;
    processing_time_seconds: number;
    python_extraction?: {
        pages: Array<{
            page_num: number;
            pymupdf_text: string;
            pdfplumber_text: string;
            pdfminer_text: string;
            tables: any[];
        }>;
    };
    ocr_extraction?: {
        pages: Array<{
            page_num: number;
            markdown_text: string;
        }>;
        error?: string;
    };
    docai_extraction?: {
        pages: Array<{
            page_num: number;
            text: string;
            tables: any[];
        }>;
        error?: string;
    };
}

function formatContextForChat(contextJson: string): string {
    try {
        const data: PipelineResult = JSON.parse(contextJson);

        let formatted = "";

        // Include ALL extraction sources so the model can cross-reference
        if (data.python_extraction?.pages) {
            formatted += "=== SOURCE 1: PYTHON TEXT EXTRACTION (PyMuPDF + pdfplumber + pdfminer.six) ===\n\n";
            for (const page of data.python_extraction.pages) {
                formatted += `--- Page ${page.page_num} ---\n`;

                // Include all three extractions so the model can cross-reference
                const pymupdf = (page.pymupdf_text || "").trim();
                const plumber = (page.pdfplumber_text || "").trim();
                const miner = (page.pdfminer_text || "").trim();

                // Pick the longest as primary
                const all = [pymupdf, plumber, miner].filter(t => t);
                all.sort((a, b) => b.length - a.length);

                if (all.length > 0) {
                    formatted += all[0] + "\n";

                    // If a shorter extraction has content not in the longest, append it
                    for (let i = 1; i < all.length; i++) {
                        const extra = all[i];
                        // Only add if it has substantial unique content (>20% different)
                        if (extra.length > 50 && extra.length > all[0].length * 0.3) {
                            formatted += `\n[Additional extraction]:\n${extra}\n`;
                        }
                    }
                }

                // Tables
                if (page.tables && page.tables.length > 0) {
                    formatted += "\nTABLES DETECTED:\n";
                    for (const table of page.tables) {
                        if (Array.isArray(table)) {
                            for (const row of table) {
                                if (Array.isArray(row)) {
                                    formatted += `  | ${row.map(c => c || "").join(" | ")} |\n`;
                                }
                            }
                            formatted += "\n";
                        }
                    }
                }
                formatted += "\n";
            }
        }

        // Mistral OCR results
        if (data.ocr_extraction?.pages && data.ocr_extraction.pages.length > 0) {
            formatted += "=== SOURCE 2: MISTRAL OCR EXTRACTION ===\n\n";
            for (const page of data.ocr_extraction.pages) {
                formatted += `--- Page ${page.page_num} ---\n`;
                formatted += (page.markdown_text || "") + "\n\n";
            }
        }

        // Google Document AI results
        if (data.docai_extraction?.pages && data.docai_extraction.pages.length > 0) {
            formatted += "=== SOURCE 3: GOOGLE DOCUMENT AI OCR ===\n\n";
            for (const page of data.docai_extraction.pages) {
                formatted += `--- Page ${page.page_num} ---\n`;
                formatted += (page.text || "") + "\n";
                if (page.tables && page.tables.length > 0) {
                    formatted += "\nTABLES DETECTED:\n";
                    for (const table of page.tables) {
                        if (Array.isArray(table)) {
                            for (const row of table) {
                                if (Array.isArray(row)) {
                                    formatted += `  | ${row.map((c: any) => c || "").join(" | ")} |\n`;
                                }
                            }
                            formatted += "\n";
                        }
                    }
                }
                formatted += "\n";
            }
        }

        // Document metadata
        formatted += "=== DOCUMENT METADATA ===\n";
        formatted += `Document Type: ${data.document_type?.replace(/_/g, " ") || "Unknown"}\n`;
        formatted += `Total Pages: ${data.total_pages || "Unknown"}\n`;

        return formatted;

    } catch {
        return contextJson || "No context provided.";
    }
}

export async function POST(req: NextRequest) {
    try {
        const { message, context, conversationHistory } = await req.json();

        if (!message) {
            return NextResponse.json({ error: "Missing message" }, { status: 400 });
        }

        const hasHistory = conversationHistory && Array.isArray(conversationHistory) && conversationHistory.length > 0;
        const formattedContext = formatContextForChat(context);

        // Only include full document context on first message to avoid token overflow
        const contextBlock = hasHistory
            ? "\n[Document context was provided in the first message above. Refer to the conversation history.]\n"
            : `\nEXTRACTED DOCUMENT DATA:\n========================\n${formattedContext}\n========================\n`;

        const messages: OpenAI.ChatCompletionMessageParam[] = [
            {
                role: "system",
                content: `You are a world-class Construction & Architecture Document Analyst. You have deep expertise across ALL types of construction documents including:
- Floor plans, site plans, roof plans
- Structural drawings (foundation, framing, reinforcement, skeleton plans)
- MEP drawings (mechanical, electrical, plumbing, HVAC)
- Interior design plans (furniture layouts, finishes, lighting)
- Elevation and section drawings
- Schedules (door, window, finish, equipment)
- Detail drawings and specifications
- Civil/landscape drawings

A document has been processed through multiple OCR extraction streams. The extracted data from independent sources is provided below. These sources may contain overlapping, complementary, or occasionally conflicting data.
${contextBlock}

CRITICAL INSTRUCTIONS FOR UNDERSTANDING THE DATA:

1. IDENTIFY WHAT EVERYTHING REPRESENTS: Don't just extract raw text — understand the MEANING and CONTEXT of every element:
   - Numbers near doors/windows = door/window tags or schedule references (e.g., "D1", "W3", "101")
   - Numbers along lines with arrows = dimensions/measurements
   - Numbers inside rooms/spaces = room numbers or area calculations
   - Circled numbers = detail references or section callouts
   - Numbers in title blocks = drawing numbers, revision numbers, scales
   - Grid lines with letters/numbers (A, B, C... or 1, 2, 3...) = structural grid references
   - Hatched/shaded areas = materials, sections, or zones
   - Dashed lines = hidden edges, overhead elements, or future work
   - Center lines = axes of symmetry
   - Arrow callouts = leader lines pointing to specific elements
   - Text near symbols = equipment tags, fixture types, material specs

2. UNDERSTAND SPATIAL RELATIONSHIPS: When text/numbers appear near each other, infer their relationship:
   - Labels adjacent to spaces = room names (e.g., "BEDROOM", "KITCHEN", "LOBBY")
   - Dimensions between walls = room sizes
   - Text near equipment symbols = equipment specifications
   - Notes with arrows = callouts explaining specific construction details
   - Column marks at grid intersections = structural column locations
   - Level markers = floor elevations (e.g., "+0.00", "FFL +3.500")

3. CROSS-REFERENCE all extraction sources. If one source captures something others missed, use the combined information.

4. Quote exact text, dimensions, labels, and values — do not paraphrase numbers or measurements. Preserve exact formats (e.g., 12'-6", 3000mm, 1.5m).

5. Be EXHAUSTIVE — scan ALL pages and ALL sources to find every relevant item.

6. Present tables in clear formatted structures.

7. If something is NOT in the extracted data, say so clearly. Do NOT fabricate or guess.

8. Flag ambiguous or partially extracted data as uncertain.

9. When making inferences (e.g., calculating area), label them as "Calculated/Inferred" and show your work.

10. NEVER discuss the OCR process or how data was obtained. Respond as if reading the original document.

RESPONSE STYLE:
- Be direct and specific. Answer first, then supporting details.
- Use bullet points for lists, tables for tabular data.
- Organize complex answers with clear headings.
- For broad questions (e.g., "summarize"), provide a structured overview: document type, project info, key spaces/elements, structural system, materials, specifications, schedules, and any special notes or details found.`
            }
        ];

        if (conversationHistory && Array.isArray(conversationHistory)) {
            for (const msg of conversationHistory as ConversationMessage[]) {
                messages.push({
                    role: msg.role,
                    content: msg.content
                });
            }
        }

        messages.push({ role: "user", content: message });

        const completion = await openai.chat.completions.create({
            model: "gpt-4o",
            messages,
            temperature: 0.2,
            max_tokens: 4096,
        });

        const responseText = completion.choices[0].message.content || "I couldn't generate a response.";

        return NextResponse.json({
            role: "assistant",
            content: responseText
        });

    } catch (error: any) {
        console.error("Chat Error:", error);
        return NextResponse.json(
            { error: "Failed to generate response", details: error.message },
            { status: 500 }
        );
    }
}
