"use client";

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Send, Bot, User, FileText, ChevronLeft, Sparkles, Copy, Check } from "lucide-react";
import { cn } from "@/lib/utils";

interface ChatInterfaceProps {
    summary: string;
    threadId: string;
    assistantId: string;
    onBack: () => void;
}

interface Message {
    role: "user" | "assistant";
    content: string;
}

export default function ChatInterface({ summary, threadId, assistantId, onBack }: ChatInterfaceProps) {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const [isCopied, setIsCopied] = useState(false);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSend = async () => {
        if (!input.trim() || isLoading) return;

        const userMessage = input;
        setInput("");
        setMessages((prev) => [...prev, { role: "user", content: userMessage }]);
        setIsLoading(true);

        try {
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    message: userMessage,
                    threadId,
                    assistantId
                }),
            });

            const data = await response.json();

            if (data.error) throw new Error(data.error);

            setMessages((prev) => [...prev, { role: "assistant", content: data.content }]);
        } catch (error) {
            console.error("Chat error:", error);
            setMessages((prev) => [
                ...prev,
                { role: "assistant", content: "Sorry, I encountered an error answering that." },
            ]);
        } finally {
            setIsLoading(false);
        }
    };

    const copySummary = () => {
        navigator.clipboard.writeText(summary);
        setIsCopied(true);
        setTimeout(() => setIsCopied(false), 2000);
    }

    return (
        <div className="flex h-[calc(100vh-4rem)] w-full max-w-6xl gap-6 p-4">
            {/* Sidebar - Summary */}
            <motion.div
                initial={{ x: -20, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                className="w-1/3 hidden md:flex flex-col gap-4"
            >
                <button
                    onClick={onBack}
                    className="flex items-center gap-2 text-sm text-white/50 hover:text-white transition-colors mb-2"
                >
                    <ChevronLeft className="w-4 h-4" />
                    Back to Upload
                </button>

                <div className="glass rounded-3xl p-6 flex-1 overflow-hidden flex flex-col">
                    <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-2 text-violet-300">
                            <FileText className="w-5 h-5" />
                            <h3 className="font-semibold">Document Analysis</h3>
                        </div>
                        <button onClick={copySummary} className="p-1 hover:bg-white/10 rounded-lg transition-colors text-white/50 hover:text-white">
                            {isCopied ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
                        </button>
                    </div>
                    <div className="overflow-y-auto pr-2 custom-scrollbar flex-1 text-sm text-white/80 leading-relaxed whitespace-pre-wrap">
                        {summary}
                    </div>
                </div>
            </motion.div>

            {/* Main Chat Area */}
            <motion.div
                initial={{ x: 20, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                className="flex-1 flex flex-col glass rounded-3xl overflow-hidden relative"
            >
                {/* Chat Header */}
                <div className="p-4 border-b border-white/10 bg-white/5 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-full bg-gradient-to-tr from-violet-500 to-fuchsia-500 flex items-center justify-center shadow-lg shadow-violet-500/20">
                            <Bot className="w-5 h-5 text-white" />
                        </div>
                        <div>
                            <h2 className="font-semibold text-white">OCR Assistant</h2>
                            <p className="text-xs text-white/50 flex items-center gap-1">
                                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                                Online & Ready
                            </p>
                        </div>
                    </div>

                    {/* Mobile Back Button */}
                    <button onClick={onBack} className="md:hidden p-2 text-white/50 hover:text-white">
                        <ChevronLeft />
                    </button>
                </div>

                {/* Messages */}
                <div className="flex-1 overflow-y-auto p-6 space-y-6 custom-scrollbar">
                    {messages.length === 0 && (
                        <div className="flex flex-col items-center justify-center h-full text-center text-white/30 space-y-4">
                            <Sparkles className="w-12 h-12 text-white/10" />
                            <div>
                                <p className="text-lg font-medium text-white/50">Ask me anything about the document</p>
                                <p className="text-sm">I can answer specific questions based on the analysis.</p>
                            </div>
                        </div>
                    )}

                    {messages.map((msg, i) => (
                        <motion.div
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            key={i}
                            className={cn(
                                "flex gap-4 max-w-[85%]",
                                msg.role === "user" ? "ml-auto flex-row-reverse" : ""
                            )}
                        >
                            <div className={cn(
                                "w-8 h-8 rounded-full flex items-center justify-center shrink-0",
                                msg.role === "user" ? "bg-white/10" : "bg-violet-500/20"
                            )}>
                                {msg.role === "user" ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4 text-violet-300" />}
                            </div>
                            <div className={cn(
                                "p-4 rounded-2xl text-sm leading-relaxed",
                                msg.role === "user"
                                    ? "bg-white/10 text-white rounded-tr-none"
                                    : "bg-gradient-to-br from-violet-500/10 to-fuchsia-500/10 border border-white/5 text-white/90 rounded-tl-none"
                            )}>
                                {msg.content}
                            </div>
                        </motion.div>
                    ))}
                    {isLoading && (
                        <div className="flex gap-4 max-w-[85%] animate-pulse">
                            <div className="w-8 h-8 rounded-full bg-violet-500/20 flex items-center justify-center shrink-0">
                                <Bot className="w-4 h-4 text-violet-300" />
                            </div>
                            <div className="p-4 rounded-2xl rounded-tl-none bg-white/5 border border-white/5">
                                <div className="flex gap-1.5">
                                    <span className="w-1.5 h-1.5 bg-white/40 rounded-full animate-bounce [animation-delay:-0.3s]" />
                                    <span className="w-1.5 h-1.5 bg-white/40 rounded-full animate-bounce [animation-delay:-0.15s]" />
                                    <span className="w-1.5 h-1.5 bg-white/40 rounded-full animate-bounce" />
                                </div>
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>

                {/* Input */}
                <div className="p-4 bg-white/5 border-t border-white/10">
                    <form
                        onSubmit={(e) => { e.preventDefault(); handleSend(); }}
                        className="flex gap-2"
                    >
                        <input
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder="Type your question..."
                            className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white placeholder:text-white/30 focus:outline-none focus:ring-1 focus:ring-violet-500/50 transition-all"
                        />
                        <button
                            type="submit"
                            disabled={isLoading || !input.trim()}
                            className="p-3 bg-violet-600 hover:bg-violet-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-xl transition-colors shadow-lg shadow-violet-600/20 text-white"
                        >
                            <Send className="w-5 h-5" />
                        </button>
                    </form>
                </div>
            </motion.div>
        </div>
    );
}
