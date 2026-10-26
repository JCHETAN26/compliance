import type { Metadata } from "next";

import { RulePacksPage } from "@/components/reviews/rule-packs-page";

export const metadata: Metadata = { title: "Rule packs" };

export default function WorkspaceRulePacksPage() {
  return <RulePacksPage />;
}
