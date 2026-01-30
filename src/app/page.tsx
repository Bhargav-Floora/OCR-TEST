"use client";

import { useState } from "react";
import UploadSection from "@/components/UploadSection";
import ChatInterface from "@/components/ChatInterface";
import { motion, AnimatePresence } from "framer-motion";

type AppState = "idle" | "analyzing" | "chat";

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

export default function Home() {
  const [appState, setAppState] = useState<AppState>("idle");
  const [summary, setSummary] = useState<string>("");
  const [pipelineResult, setPipelineResult] = useState<PipelineResult | undefined>();

  const handleFileSelect = async (file: File) => {
    setAppState("analyzing");

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
      setPipelineResult(data.pipelineResult);
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
    setPipelineResult(undefined);
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
              onBack={handleBack}
              pipelineResult={pipelineResult}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
