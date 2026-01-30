import { NextRequest, NextResponse } from "next/server";
import { spawn } from "child_process";
import path from "path";
import fs from "fs";
import os from "os";

function setupGoogleCredentials(): void {
    const credJson = process.env.GOOGLE_SERVICE_ACCOUNT_KEY;
    if (credJson && !process.env.GOOGLE_APPLICATION_CREDENTIALS) {
        const tmpPath = path.join(os.tmpdir(), "gcp-service-account.json");
        fs.writeFileSync(tmpPath, credJson);
        process.env.GOOGLE_APPLICATION_CREDENTIALS = tmpPath;
    }
}

export async function POST(req: NextRequest) {
    try {
        setupGoogleCredentials();

        const formData = await req.formData();
        const file = formData.get("file") as File;

        if (!file) {
            return NextResponse.json({ error: "No file uploaded" }, { status: 400 });
        }

        const fileSizeMB = (file.size / (1024 * 1024)).toFixed(2);
        console.log(`\n${"=".repeat(60)}`);
        console.log(`Processing: ${file.name} (${fileSizeMB} MB)`);
        console.log(`Starting Dual-Stream OCR Pipeline...`);
        console.log(`${"=".repeat(60)}\n`);

        const arrayBuffer = await file.arrayBuffer();
        const buffer = Buffer.from(arrayBuffer);

        const scriptPath = path.join(process.cwd(), "scripts", "ocr_pipeline.py");
        const pythonExec = process.env.PYTHON_PATH || path.join(process.cwd(), "venv", "bin", "python");
        const pythonProcess = spawn(pythonExec, [scriptPath], {
            env: {
                ...process.env,
                PYTHONPATH: process.cwd(),
            }
        });

        let outputData = "";
        let errorData = "";

        pythonProcess.stdin.write(buffer);
        pythonProcess.stdin.end();

        return new Promise<NextResponse>((resolve) => {
            pythonProcess.stdout.on("data", (data) => {
                outputData += data.toString();
            });

            pythonProcess.stderr.on("data", (data) => {
                const message = data.toString().trim();
                if (message) {
                    console.log(`[Pipeline]: ${message}`);
                }
                errorData += data.toString();
            });

            pythonProcess.on("close", (code) => {
                console.log(`\nPipeline completed with exit code: ${code}\n`);

                if (code !== 0) {
                    console.error(`Pipeline error details:\n${errorData}`);
                    try {
                        const jsonError = JSON.parse(outputData);
                        resolve(NextResponse.json(jsonError, { status: 400 }));
                    } catch {
                        resolve(NextResponse.json(
                            { error: "Analysis failed", details: errorData || "Unknown pipeline error" },
                            { status: 500 }
                        ));
                    }
                    return;
                }

                try {
                    const result = JSON.parse(outputData);
                    if (result.error) {
                        resolve(NextResponse.json(result, { status: 400 }));
                        return;
                    }

                    console.log(`Document Type: ${result.document_type}`);
                    console.log(`Pages: ${result.total_pages}`);
                    console.log(`Time: ${result.processing_time_seconds}s`);

                    resolve(NextResponse.json({
                        analysis: JSON.stringify(result),
                        pipelineResult: result,
                    }));
                } catch (e: any) {
                    console.error("Failed to parse pipeline output:", outputData.substring(0, 500));
                    resolve(NextResponse.json(
                        { error: "Invalid response from pipeline", details: e.message },
                        { status: 500 }
                    ));
                }
            });
        });

    } catch (error: any) {
        console.error("Analysis Error:", error);
        return NextResponse.json(
            { error: "Analysis failed", details: error.message },
            { status: 500 }
        );
    }
}
