"use client";

import { useQuery } from "@tanstack/react-query";
import { FlaskConical, TriangleAlert } from "lucide-react";

import { fetchModelStatus, modelKeys } from "@/lib/api/queries";
import { cn } from "@/lib/utils";


/**
 * Site-wide notice of the rule-check model's actual availability (handoff section 7.8:
 * without trained weights the app still runs, and the UI says so). Shows nothing when a
 * trained model is available or while the status is loading. The text comes from the API.
 */
export function ModelStatusBanner() {
  const status = useQuery({
    queryKey: modelKeys.status,
    queryFn: fetchModelStatus,
    staleTime: 60_000,
    refetchInterval: 120_000,
  });
  const ruleCheck = status.data?.rule_check;
  // A trained model needs no notice unless reviews deliberately use the mock backend.
  if (!ruleCheck || (ruleCheck.state === "trained" && status.data?.review_rule_check_backend !== "mock")) return null;
  const severe = ruleCheck.state === "smoke";
  const Icon = severe ? FlaskConical : TriangleAlert;
  return (
    <div
      role="status"
      aria-live="polite"
      data-state={ruleCheck.state}
      className={cn(
        "border-b px-4 py-2 text-sm sm:px-6",
        severe
          ? "border-destructive/40 bg-destructive/10 text-destructive"
          : "border-amber-300 bg-amber-50 text-amber-900 dark:border-amber-700 dark:bg-amber-950 dark:text-amber-100",
      )}
    >
      <div className="mx-auto flex max-w-6xl items-start gap-2">
        <Icon className="mt-0.5 size-4 shrink-0" aria-hidden />
        <p>
          {ruleCheck.message}
          {ruleCheck.model_version ? (
            <span className="ml-1 font-mono text-xs opacity-80">({ruleCheck.model_version})</span>
          ) : null}
        </p>
      </div>
    </div>
  );
}
