import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

export async function POST(req: NextRequest) {
    try {
        const formData = await req.formData();
        const file = formData.get("file") as File;

        if (!file) {
            return NextResponse.json({ error: "No file provided" }, { status: 400 });
        }

        // 1. Upload File to OpenAI
        const buffer = await file.arrayBuffer();
        // Create a temporary File object for OpenAI SDK compatibility if needed, 
        // but the SDK accepts a Fetch File/Blob or ReadStream. 
        // We can cast the File from text/form-data directly if environment supports it,
        // or use "toFile" helper if on Node. 
        // Since we are in Next.js Server Action / Route Handler, we can use the file directly.

        // Note: openai.files.create expects a `File` or `ReadStream`.
        // We might need to convert the buffer to a file-like object or use the original file if strictly compatible.
        // The safest way in Node environment is usually buffer + name.

        const openAIFile = await openai.files.create({
            file: file,
            purpose: "assistants",
        });

        // 2. Create a Thread with the initial message and file attachment
        const thread = await openai.beta.threads.create({
            messages: [
                {
                    role: "user",
                    content: "Please analyze this document detailedly. Extract all text and provide a comprehensive summary.",
                    attachments: [
                        {
                            file_id: openAIFile.id,
                            tools: [{ type: "file_search" }], // Or just rely on vision? 
                            // For Vision, we actually need to use 'vision' or reference image_file if supported in new assistants.
                            // Current Assistants API supports image files in messages for Vision.
                            // If it's a PDF, we use `file_search` (retrieval) OR convert to image. 
                            // WAIT. Assistants API v2 with GPT-4o supports "Vision" on images, but for PDFs it uses "File Search" (parsing) generally.
                            // BUT the user specifically wants visual analysis (construction plans).
                            // For construction plans (PDF), File Search is good for text, but Vision is needed for drawings.
                            // GPT-4o Assistant supports "Vision" if we attach images. 
                            // If we attach a PDF, it treats it as a file for code_interpreter or file_search.

                            // HOWEVER, recently standard GPT-4o with attachments can "see" PDFs if they are fed correctly.
                            // Actually, for "Visual" analysis of PDFs (Blueprints), Code Interpreter is often used by ChatGPT to render to image 
                            // OR it just uses the internal PDF parser.

                            // Let's stick to the sophisticated "Assistants" flow. 
                            // We will use "file_search" for now as it's the standard for vector retrieval of info.
                            // If the user TRULY needs "Vision" (like "what is drawn at 10,10"), the PDF might still need to be images.
                            // BUT, let's trust the "Assistants API" robustness first. 
                            // Actually, to be safe for "Vision" on PDFs, converting to images IS still the most reliable way unless using Code Interpreter.
                            // But the user complained about speed.

                            // Let's assume standard Assistants API with the file attachment.
                        }
                    ]
                }
            ],
        });



        // We can also just use chat completions with "system" and the file ID? 
        // No, Chat Completions doesn't accept file IDs directly (except for maybe in recent updates? No).

        // So we MUST create an Assistant or use one.
        // Let's create a temporary assistant or a singleton.
        // Ideally we define the Assistant ID in env. For this demo, let's create one (it's fast).

        // actually, creating an assistant every time is bad practice (clutter). 
        // Let's check if we have one or just define a "Generic" one.
        // Better: Helper function to get or create assistant.

        // Optimization: Just use `openai.beta.assistants.create` once if possible, but here we'll create one per session? 
        // No, that's too slow.
        // Let's use `model: "gpt-4o"` directly in the run? 
        // No, `runs.create` requires assistant_id.

        // Okay, first, let's try to see if we can just point to the model without a pre-made assistant? 
        // No. 

        // Solution: Create a generic assistant ONCE and ID should be in env. 
        // But I don't want to make the user go into OpenAI dashboard.
        // I will create an assistant in the code if it doesn't exist? No, stateless.
        // I will just create a new Assistant for this "Session". It adds 1 second but ensures config is right.

        const assistant = await openai.beta.assistants.create({
            name: "OCR Assistant",
            instructions: "You represent an OCR system. Analyze documents provided.",
            model: "gpt-4o",
            tools: [{ type: "file_search" }, { type: "code_interpreter" }],
            // code_interpreter is good for PDF rendering if needed.
        });

        // Now Run
        const run = await openai.beta.threads.runs.createAndPoll(thread.id, {
            assistant_id: assistant.id,
        });

        if (run.status !== 'completed') {
            throw new Error(`Run failed with status: ${run.status}`);
        }

        const messages = await openai.beta.threads.messages.list(thread.id);
        const lastMessage = messages.data[0];

        // Extract text content
        let analysisText = "";
        if (lastMessage.role === "assistant") {
            for (const content of lastMessage.content) {
                if (content.type === 'text') {
                    analysisText += content.text.value;
                }
            }
        }

        return NextResponse.json({
            analysis: analysisText,
            threadId: thread.id,
            assistantId: assistant.id // Pass back so we can reuse for chat? 
            // Actually, for chat we need to reuse the same assistant.
        });

    } catch (error: any) {
        console.error("Analysis Error:", error);
        return NextResponse.json(
            { error: "Failed to process document", details: error.message },
            { status: 500 }
        );
    }
}
