import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "OCR Studio | AI-Powered Document Analysis",
  description: "Upload detailed documents for instant OCR and interactive Q&A powered by GPT-4o.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased min-h-screen text-foreground selection:bg-primary/30 selection:text-white`}
      >
        <main className="relative z-10 flex flex-col items-center justify-center min-h-screen p-4 sm:p-8">
          {children}
        </main>
      </body>
    </html>
  );
}
