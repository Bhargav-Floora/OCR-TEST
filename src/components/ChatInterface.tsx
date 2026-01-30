"use client";

import { useState, useRef, useEffect, useMemo } from "react";
import { motion } from "framer-motion";
import { Send, Bot, User, FileText, ChevronLeft, Copy, Check, Clock } from "lucide-react";
import { cn } from "@/lib/utils";

interface PipelineResult {
    document_type: string;
    total_pages: number;
    processing_time_seconds: number;
    python_extraction?: {
        pages: Array<{
            page_num: number;
            pymupdf_text: string;
            pymupdf_blocks: any[];
            pdfplumber_text: string;
            pdfplumber_words: any[];
            pdfminer_text: string;
            pdfminer_elements: any[];
            tables: any[];
            drawings_count: number;
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

interface ChatInterfaceProps {
    summary: string;
    onBack: () => void;
    pipelineResult?: PipelineResult;
}

interface Message {
    role: "user" | "assistant";
    content: string;
}

type ActiveTab = "python" | "ocr" | "docai";

export default function ChatInterface({ summary, onBack, pipelineResult }: ChatInterfaceProps) {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const [isCopied, setIsCopied] = useState(false);
    const [activeTab, setActiveTab] = useState<ActiveTab>("python");

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const parsedResult = useMemo(() => {
        if (pipelineResult) return pipelineResult;
        try {
            return JSON.parse(summary) as PipelineResult;
        } catch {
            return null;
        }
    }, [summary, pipelineResult]);

    const handleSend = async () => {
        if (!input.trim() || isLoading) return;

        const userMessage = input;
        setInput("");
        const newMessages: Message[] = [...messages, { role: "user", content: userMessage }];
        setMessages(newMessages);
        setIsLoading(true);

        try {
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    message: userMessage,
                    context: summary,
                    conversationHistory: messages,
                }),
            });

            const data = await response.json();
            if (data.error) {
                setMessages([...newMessages, { role: "assistant", content: `Error: ${data.error}` }]);
            } else {
                setMessages([...newMessages, { role: "assistant", content: data.content }]);
            }
        } catch {
            setMessages([...newMessages, { role: "assistant", content: "Failed to get response. Please try again." }]);
        } finally {
            setIsLoading(false);
        }
    };

    const copySummary = () => {
        navigator.clipboard.writeText(summary);
        setIsCopied(true);
        setTimeout(() => setIsCopied(false), 2000);
    };

    // Combine all Python extraction text for display
    const getPythonText = (page: any): string => {
        const texts = [
            page.pymupdf_text,
            page.pdfplumber_text,
            page.pdfminer_text,
        ].filter((t: string) => t && t.trim());

        // Return longest (most complete) extraction
        if (texts.length === 0) return "No text extracted";
        return texts.sort((a: string, b: string) => b.length - a.length)[0];
    };

    return (
        <div className="w-full max-w-7xl mx-auto p-4 flex gap-4 h-[calc(100vh-2rem)]">
            {/* Left Panel - Extracted Data */}
            <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                className="w-96 flex flex-col"
            >
                <button
                    onClick={onBack}
                    className="flex items-center gap-2 text-sm text-white/50 hover:text-white transition-colors mb-2"
                >
                    <ChevronLeft className="w-4 h-4" />
                    Back to Upload
                </button>

                <div className="glass rounded-3xl p-4 flex-1 overflow-hidden flex flex-col">
                    <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-2 text-violet-300">
                            <FileText className="w-5 h-5" />
                            <h3 className="font-semibold">Extracted Data</h3>
                        </div>
                        <button onClick={copySummary} className="p-1 hover:bg-white/10 rounded-lg transition-colors text-white/50 hover:text-white">
                            {isCopied ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
                        </button>
                    </div>

                    {/* Stats Bar */}
                    {parsedResult && (
                        <div className="flex items-center gap-3 text-xs text-white/40 mb-3 pb-2 border-b border-white/10">
                            <span className="flex items-center gap-1">
                                <Clock className="w-3 h-3" />
                                {parsedResult.processing_time_seconds}s
                            </span>
                            <span>{parsedResult.total_pages} page(s)</span>
                            <span className="text-white/30">{parsedResult.document_type?.replace(/_/g, " ")}</span>
                        </div>
                    )}

                    {/* Tabs */}
                    <div className="flex gap-1 mb-3">
                        <button
                            onClick={() => setActiveTab("python")}
                            className={cn(
                                "px-3 py-1.5 text-xs rounded-lg transition-colors",
                                activeTab === "python"
                                    ? "bg-emerald-500/20 text-emerald-300"
                                    : "bg-white/5 text-white/40 hover:text-white/60"
                            )}
                        >
                            Python Extraction
                        </button>
                        <button
                            onClick={() => setActiveTab("ocr")}
                            className={cn(
                                "px-3 py-1.5 text-xs rounded-lg transition-colors",
                                activeTab === "ocr"
                                    ? "bg-violet-500/20 text-violet-300"
                                    : "bg-white/5 text-white/40 hover:text-white/60"
                            )}
                        >
                            Mistral OCR
                        </button>
                        <button
                            onClick={() => setActiveTab("docai")}
                            className={cn(
                                "px-3 py-1.5 text-xs rounded-lg transition-colors",
                                activeTab === "docai"
                                    ? "bg-amber-500/20 text-amber-300"
                                    : "bg-white/5 text-white/40 hover:text-white/60"
                            )}
                        >
                            Document AI
                        </button>
                    </div>

                    {/* Content */}
                    <div className="overflow-y-auto pr-2 custom-scrollbar flex-1">
                        {activeTab === "docai" ? (
                            <div className="space-y-3">
                                {parsedResult?.docai_extraction?.error && (
                                    <div className="p-2 rounded-lg bg-red-500/10 border border-red-500/20 text-xs text-red-300">
                                        Document AI Error: {parsedResult.docai_extraction.error}
                                    </div>
                                )}
                                {parsedResult?.docai_extraction?.pages?.map((page) => (
                                    <div key={page.page_num} className="space-y-2">
                                        <h4 className="text-xs font-bold text-amber-400 uppercase">
                                            Page {page.page_num}
                                        </h4>
                                        <div className="p-2 rounded-lg bg-white/5 border border-white/10 max-h-64 overflow-y-auto">
                                            <pre className="text-xs text-white/80 whitespace-pre-wrap break-words font-mono">
                                                {page.text || "No text extracted"}
                                            </pre>
                                        </div>
                                        {page.tables && page.tables.length > 0 && (
                                            <div>
                                                <h5 className="text-[10px] font-bold text-blue-400 uppercase mb-1">
                                                    Tables ({page.tables.length})
                                                </h5>
                                                {page.tables.map((table: any, ti: number) => (
                                                    <div key={ti} className="p-2 rounded-lg bg-blue-500/10 border border-blue-500/20 mb-1 overflow-x-auto">
                                                        <table className="text-[10px] text-white/70">
                                                            <tbody>
                                                                {Array.isArray(table) && table.map((row: any, ri: number) => (
                                                                    <tr key={ri}>
                                                                        {Array.isArray(row) && row.map((cell: any, ci: number) => (
                                                                            <td key={ci} className="px-1 py-0.5 border border-white/10">
                                                                                {cell || ""}
                                                                            </td>
                                                                        ))}
                                                                    </tr>
                                                                ))}
                                                            </tbody>
                                                        </table>
                                                    </div>
                                                ))}
                                            </div>
                                        )}
                                        <div className="text-[10px] text-white/30">
                                            {(page.text || "").length} chars
                                        </div>
                                    </div>
                                )) || (
                                    <div className="text-white/40 text-center py-4 text-sm">
                                        No Document AI data
                                    </div>
                                )}
                            </div>
                        ) : activeTab === "python" ? (
                            <div className="space-y-3">
                                {parsedResult?.python_extraction?.pages?.map((page) => (
                                    <div key={page.page_num} className="space-y-2">
                                        <h4 className="text-xs font-bold text-emerald-400 uppercase">
                                            Page {page.page_num}
                                        </h4>

                                        {/* Raw text */}
                                        <div className="p-2 rounded-lg bg-white/5 border border-white/10 max-h-64 overflow-y-auto">
                                            <pre className="text-xs text-white/80 whitespace-pre-wrap break-words font-mono">
                                                {getPythonText(page)}
                                            </pre>
                                        </div>

                                        {/* Tables */}
                                        {page.tables && page.tables.length > 0 && (
                                            <div>
                                                <h5 className="text-[10px] font-bold text-blue-400 uppercase mb-1">
                                                    Tables ({page.tables.length})
                                                </h5>
                                                {page.tables.map((table: any, ti: number) => (
                                                    <div key={ti} className="p-2 rounded-lg bg-blue-500/10 border border-blue-500/20 mb-1 overflow-x-auto">
                                                        <table className="text-[10px] text-white/70">
                                                            <tbody>
                                                                {Array.isArray(table) && table.map((row: any, ri: number) => (
                                                                    <tr key={ri}>
                                                                        {Array.isArray(row) && row.map((cell: any, ci: number) => (
                                                                            <td key={ci} className="px-1 py-0.5 border border-white/10">
                                                                                {cell || ""}
                                                                            </td>
                                                                        ))}
                                                                    </tr>
                                                                ))}
                                                            </tbody>
                                                        </table>
                                                    </div>
                                                ))}
                                            </div>
                                        )}

                                        {/* Extraction source details */}
                                        <div className="flex flex-wrap gap-1 text-[10px] text-white/30">
                                            <span>PyMuPDF: {(page.pymupdf_text || "").length} chars</span>
                                            <span>|</span>
                                            <span>pdfplumber: {(page.pdfplumber_text || "").length} chars</span>
                                            <span>|</span>
                                            <span>pdfminer: {(page.pdfminer_text || "").length} chars</span>
                                            <span>|</span>
                                            <span>{page.drawings_count} drawings</span>
                                        </div>
                                    </div>
                                )) || (
                                    <div className="text-white/40 text-center py-4 text-sm">
                                        No Python extraction data
                                    </div>
                                )}
                            </div>
                        ) : (
                            <div className="space-y-3">
                                {parsedResult?.ocr_extraction?.error && (
                                    <div className="p-2 rounded-lg bg-red-500/10 border border-red-500/20 text-xs text-red-300">
                                        OCR Error: {parsedResult.ocr_extraction.error}
                                    </div>
                                )}
                                {parsedResult?.ocr_extraction?.pages?.map((page) => (
                                    <div key={page.page_num} className="space-y-2">
                                        <h4 className="text-xs font-bold text-violet-400 uppercase">
                                            Page {page.page_num}
                                        </h4>
                                        <div className="p-2 rounded-lg bg-white/5 border border-white/10 max-h-64 overflow-y-auto">
                                            <pre className="text-xs text-white/80 whitespace-pre-wrap break-words font-mono">
                                                {page.markdown_text || "No text extracted"}
                                            </pre>
                                        </div>
                                        <div className="text-[10px] text-white/30">
                                            {(page.markdown_text || "").length} chars
                                        </div>
                                    </div>
                                )) || (
                                    <div className="text-white/40 text-center py-4 text-sm">
                                        No Mistral OCR data
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            </motion.div>

            {/* Right Panel - Chat */}
            <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                className="flex-1 flex flex-col glass rounded-3xl overflow-hidden"
            >
                {/* Chat Header */}
                <div className="p-4 border-b border-white/10">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-full bg-gradient-to-r from-violet-500 to-emerald-500 flex items-center justify-center">
                            <Bot className="w-6 h-6 text-white" />
                        </div>
                        <div>
                            <h2 className="font-semibold text-white">Document Assistant</h2>
                            <p className="text-xs text-white/50">Ask questions about the extracted data</p>
                        </div>
                    </div>
                </div>

                {/* Messages */}
                <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar">
                    {messages.length === 0 && (
                        <div className="text-center py-12">
                            <Bot className="w-16 h-16 mx-auto text-white/20 mb-4" />
                            <p className="text-white/40 text-sm">
                                Ask any question about the document data shown on the left.
                            </p>
                            <div className="mt-4 flex flex-wrap justify-center gap-2">
                                {["What rooms are shown?", "List all dimensions", "Summarize the document"].map((q) => (
                                    <button
                                        key={q}
                                        onClick={() => setInput(q)}
                                        className="px-3 py-1.5 text-xs bg-white/5 hover:bg-white/10 rounded-full text-white/60 hover:text-white transition-colors"
                                    >
                                        {q}
                                    </button>
                                ))}
                            </div>
                        </div>
                    )}

                    {messages.map((msg, i) => (
                        <div key={i} className={cn("flex gap-3", msg.role === "user" ? "justify-end" : "justify-start")}>
                            {msg.role === "assistant" && (
                                <div className="w-8 h-8 rounded-full bg-gradient-to-r from-violet-500 to-emerald-500 flex items-center justify-center flex-shrink-0">
                                    <Bot className="w-4 h-4 text-white" />
                                </div>
                            )}
                            <div className={cn(
                                "max-w-[70%] p-3 rounded-2xl text-sm",
                                msg.role === "user"
                                    ? "bg-violet-500/20 text-white"
                                    : "bg-white/5 text-white/90"
                            )}>
                                <div className="whitespace-pre-wrap">{msg.content}</div>
                            </div>
                            {msg.role === "user" && (
                                <div className="w-8 h-8 rounded-full bg-white/10 flex items-center justify-center flex-shrink-0">
                                    <User className="w-4 h-4 text-white" />
                                </div>
                            )}
                        </div>
                    ))}

                    {isLoading && (
                        <div className="flex gap-3">
                            <div className="w-8 h-8 rounded-full bg-gradient-to-r from-violet-500 to-emerald-500 flex items-center justify-center">
                                <Bot className="w-4 h-4 text-white" />
                            </div>
                            <div className="bg-white/5 p-3 rounded-2xl">
                                <div className="flex gap-1">
                                    <div className="w-2 h-2 bg-white/40 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                                    <div className="w-2 h-2 bg-white/40 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                                    <div className="w-2 h-2 bg-white/40 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                                </div>
                            </div>
                        </div>
                    )}

                    <div ref={messagesEndRef} />
                </div>

                {/* Input */}
                <div className="p-4 border-t border-white/10">
                    <form onSubmit={(e) => { e.preventDefault(); handleSend(); }} className="flex gap-2">
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder="Ask about the document..."
                            className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white placeholder-white/30 focus:outline-none focus:border-violet-500/50 transition-colors"
                        />
                        <button
                            type="submit"
                            disabled={!input.trim() || isLoading}
                            className={cn(
                                "p-3 rounded-xl transition-all",
                                input.trim() && !isLoading
                                    ? "bg-gradient-to-r from-violet-500 to-emerald-500 text-white hover:shadow-lg hover:shadow-violet-500/25"
                                    : "bg-white/5 text-white/30"
                            )}
                        >
                            <Send className="w-5 h-5" />
                        </button>
                    </form>
                </div>
            </motion.div>
        </div>
    );
}
