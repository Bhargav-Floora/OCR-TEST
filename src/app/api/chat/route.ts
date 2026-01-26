import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

export async function POST(req: NextRequest) {
    try {
        const { message, threadId, assistantId } = await req.json();

        if (!message || !threadId || !assistantId) {
            return NextResponse.json({ error: "Missing required fields" }, { status: 400 });
        }

        // Add message to thread
        await openai.beta.threads.messages.create(threadId, {
            role: "user",
            content: message,
        });

        // Run
        const run = await openai.beta.threads.runs.createAndPoll(threadId, {
            assistant_id: assistantId,
        });

        if (run.status !== 'completed') {
            throw new Error(`Run failed: ${run.status}`);
        }

        const messages = await openai.beta.threads.messages.list(threadId);

        // Get the latest assistant message
        // List returns newest first
        const lastMsg = messages.data.find(m => m.role === "assistant");

        let responseText = "";
        if (lastMsg) {
            for (const c of lastMsg.content) {
                if (c.type === 'text') {
                    responseText += c.text.value;
                }
            }
        }

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
