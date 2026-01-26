"use client";

import { useState } from "react";
import UploadSection from "@/components/UploadSection";
import ChatInterface from "@/components/ChatInterface";
import { motion, AnimatePresence } from "framer-motion";

type AppState = "idle" | "analyzing" | "chat";

export default function Home() {
  const [appState, setAppState] = useState<AppState>("idle");
  const [summary, setSummary] = useState<string>("");
  const [threadId, setThreadId] = useState<string>("");
  const [assistantId, setAssistantId] = useState<string>("");

  const handleFileSelect = async (file: File) => {
    setAppState("analyzing");

    // Create FormData
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.details || errorData.error || "Analysis failed");
      }

      const data = await res.json();
      setSummary(data.analysis);
      setThreadId(data.threadId);
      setAssistantId(data.assistantId);
      setAppState("chat");
    } catch (error) {
      console.error(error);
      alert(error instanceof Error ? error.message : "Failed to analyze document. Please try again.");
      setAppState("idle");
    }
  };

  const handleBack = () => {
    setAppState("idle");
    setSummary("");
    setThreadId("");
    setAssistantId("");
  };

  return (
    <div className="w-full h-full flex flex-col items-center justify-center">
      <AnimatePresence mode="wait">
        {appState === "idle" || appState === "analyzing" ? (
          <motion.div
            key="upload"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="w-full"
          >
            <UploadSection
              onFileSelect={handleFileSelect}
              isAnalyzing={appState === "analyzing"}
            />
          </motion.div>
        ) : (
          <motion.div
            key="chat"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="w-full flex justify-center"
          >
            <ChatInterface
              summary={summary}
              threadId={threadId}
              assistantId={assistantId}
              onBack={handleBack}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
