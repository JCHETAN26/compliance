import type { Metadata } from "next";

import { WorkspaceSearch } from "@/components/search/workspace-search";

export const metadata: Metadata = { title: "Search" };

export default function WorkspaceSearchPage() {
  return <WorkspaceSearch />;
}
