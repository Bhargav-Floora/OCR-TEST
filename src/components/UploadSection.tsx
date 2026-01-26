"use client";

import { useState, useRef, ChangeEvent, DragEvent } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, FileText, Loader2, CheckCircle2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface UploadSectionProps {
    onFileSelect: (file: File) => void;
    isAnalyzing: boolean;
}

export default function UploadSection({ onFileSelect, isAnalyzing }: UploadSectionProps) {
    const [isDragging, setIsDragging] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        setIsDragging(false);
    };

    const handleDrop = (e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        setIsDragging(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            onFileSelect(e.dataTransfer.files[0]);
        }
    };

    const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            onFileSelect(e.target.files[0]);
        }
    };

    return (
        <div className="w-full max-w-2xl mx-auto p-6">
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-center mb-8"
            >
                <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-violet-400 to-emerald-400 mb-2">
                    OCR Studio
                </h1>
                <p className="text-white/60">Upload any document to analyze and chat instantly.</p>
            </motion.div>

            <motion.div
                onClick={() => !isAnalyzing && fileInputRef.current?.click()}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={cn(
                    "relative group glass p-12 rounded-3xl border-2 border-dashed cursor-pointer transition-all duration-300",
                    isDragging
                        ? "border-emerald-500/50 bg-emerald-500/10 scale-[1.02]"
                        : "border-white/10 hover:border-violet-500/30 hover:bg-white/5",
                    isAnalyzing && "pointer-events-none opacity-80"
                )}
                whileHover={{ scale: isAnalyzing ? 1 : 1.01 }}
                whileTap={{ scale: isAnalyzing ? 1 : 0.98 }}
            >
                <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/png, image/jpeg, image/webp, application/pdf"
                    className="hidden"
                    onChange={handleFileChange}
                />

                <div className="flex flex-col items-center justify-center gap-4 text-center">
                    <AnimatePresence mode="wait">
                        {isAnalyzing ? (
                            <motion.div
                                key="analyzing"
                                initial={{ scale: 0.8, opacity: 0 }}
                                animate={{ scale: 1, opacity: 1 }}
                                exit={{ scale: 0.8, opacity: 0 }}
                                className="flex flex-col items-center gap-4"
                            >
                                <div className="relative">
                                    <div className="absolute inset-0 rounded-full blur-xl bg-violet-500/30 animate-pulse" />
                                    <Loader2 className="w-16 h-16 text-violet-400 animate-spin relative z-10" />
                                </div>
                                <div>
                                    <h3 className="text-xl font-medium text-white mb-1">Analyzing Document...</h3>
                                    <p className="text-sm text-white/50">Using GPT-4o Vision to extract details</p>
                                </div>
                            </motion.div>
                        ) : (
                            <motion.div
                                key="upload"
                                initial={{ scale: 0.8, opacity: 0 }}
                                animate={{ scale: 1, opacity: 1 }}
                                exit={{ scale: 0.8, opacity: 0 }}
                                className="flex flex-col items-center gap-4"
                            >
                                <div className="relative w-20 h-20 rounded-full bg-white/5 flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
                                    <Upload className="w-8 h-8 text-white/70 group-hover:text-violet-400 transition-colors" />
                                </div>
                                <div>
                                    <h3 className="text-xl font-medium text-white mb-2">
                                        Click to upload or drag and drop
                                    </h3>
                                    <p className="text-sm text-white/50">
                                        Supports Images (JPG, PNG) and PDF documents
                                    </p>
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>
            </motion.div>

            {/* Footer Info */}
            <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.5 }}
                className="mt-8 flex justify-center gap-6 text-xs text-white/30"
            >
                <div className="flex items-center gap-1">
                    <CheckCircle2 className="w-3 h-3 text-emerald-500" />
                    <span>Secure Analysis</span>
                </div>
                <div className="flex items-center gap-1">
                    <CheckCircle2 className="w-3 h-3 text-emerald-500" />
                    <span>GPT-4o Powered</span>
                </div>
            </motion.div>
        </div>
    );
}
