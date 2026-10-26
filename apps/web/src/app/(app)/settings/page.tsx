import type { Metadata } from "next";

import { ApiTokensPanel } from "@/components/settings/api-tokens-panel";

export const metadata: Metadata = { title: "Settings" };

export default function UserSettingsPage() {
  return (
    <div className="grid gap-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Settings</h1>
        <p className="text-sm text-muted-foreground">
          Personal settings. Review defaults (LLM provider, threshold, budget) are set per workspace on each
          workspace&apos;s settings page.
        </p>
      </div>
      <ApiTokensPanel />
    </div>
  );
}
