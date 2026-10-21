"use client";

import { AlertTriangle } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";

export default function AppError({ reset }: { error: Error; reset: () => void }) {
  return (
    <div className="mx-auto max-w-xl py-16">
      <Alert variant="destructive">
        <AlertTriangle aria-hidden />
        <AlertTitle>This page failed to load</AlertTitle>
        <AlertDescription>
          The server could not complete the request. Check that the API is running, then retry.
          <div className="mt-3">
            <Button variant="outline" size="sm" onClick={reset}>
              Retry
            </Button>
          </div>
        </AlertDescription>
      </Alert>
    </div>
  );
}
