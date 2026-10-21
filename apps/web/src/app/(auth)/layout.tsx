import { ShieldCheck } from "lucide-react";
import type { ReactNode } from "react";

export default function AuthLayout({ children }: { children: ReactNode }) {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center gap-8 p-6">
      <div className="flex items-center gap-2 text-lg font-semibold">
        <ShieldCheck className="size-6 text-primary" aria-hidden />
        Compliance Workspace
      </div>
      {children}
    </main>
  );
}
