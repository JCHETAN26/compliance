"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useId, useState, type FormEvent } from "react";

import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { api, errorMessage, unwrap } from "@/lib/api/client";
import { safeNextPath } from "@/lib/redirect";

export const PASSWORD_MIN_LENGTH = 8;
export const PASSWORD_MAX_LENGTH = 128;

type Mode = "login" | "signup";

const COPY: Record<Mode, { title: string; description: string; submit: string }> = {
  login: {
    title: "Sign in",
    description: "Sign in to review documents in your workspaces.",
    submit: "Sign in",
  },
  signup: {
    title: "Create your account",
    description: "Start a workspace for contract and vendor compliance review.",
    submit: "Create account",
  },
};

export function validateCredentials(
  mode: Mode,
  email: string,
  password: string,
): { email?: string; password?: string } {
  const errors: { email?: string; password?: string } = {};
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim())) errors.email = "Enter a valid email address.";
  if (mode === "signup" && password.length < PASSWORD_MIN_LENGTH) {
    errors.password = `Use at least ${PASSWORD_MIN_LENGTH} characters.`;
  } else if (password.length === 0) {
    errors.password = "Enter your password.";
  } else if (password.length > PASSWORD_MAX_LENGTH) {
    errors.password = `Use at most ${PASSWORD_MAX_LENGTH} characters.`;
  }
  return errors;
}

export function AuthForm({
  mode,
  next,
  notice,
}: {
  mode: Mode;
  next?: string | null;
  notice?: string | null;
}) {
  const router = useRouter();
  const queryClient = useQueryClient();
  const id = useId();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fieldErrors, setFieldErrors] = useState<{ email?: string; password?: string }>({});
  const copy = COPY[mode];
  const destination = safeNextPath(next);

  const mutation = useMutation({
    mutationFn: async () => {
      const body = { email: email.trim(), password };
      if (mode === "login") {
        return unwrap(await api.POST("/api/v1/auth/login", { body }), "/api/v1/auth/login");
      }
      return unwrap(await api.POST("/api/v1/auth/register", { body }), "/api/v1/auth/register");
    },
    onSuccess: () => {
      queryClient.clear();
      router.replace(destination as Parameters<typeof router.replace>[0]);
      router.refresh();
    },
  });

  function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const errors = validateCredentials(mode, email, password);
    setFieldErrors(errors);
    if (Object.keys(errors).length === 0) mutation.mutate();
  }

  const emailError = fieldErrors.email;
  const passwordError = fieldErrors.password;

  return (
    <Card className="w-full max-w-sm">
      <CardHeader>
        <h1 className="text-xl leading-none font-semibold">{copy.title}</h1>
        <CardDescription>{copy.description}</CardDescription>
      </CardHeader>
      <CardContent>
        <form noValidate onSubmit={onSubmit} className="grid gap-4" aria-describedby={`${id}-status`}>
          <div id={`${id}-status`} aria-live="polite">
            {notice && !mutation.isError ? (
              <Alert>
                <AlertDescription>{notice}</AlertDescription>
              </Alert>
            ) : null}
            {mutation.isError ? (
              <Alert variant="destructive">
                <AlertDescription>{errorMessage(mutation.error)}</AlertDescription>
              </Alert>
            ) : null}
          </div>
          <div className="grid gap-2">
            <Label htmlFor={`${id}-email`}>Email</Label>
            <Input
              id={`${id}-email`}
              type="email"
              autoComplete="email"
              inputMode="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              aria-invalid={emailError ? true : undefined}
              aria-describedby={emailError ? `${id}-email-error` : undefined}
              autoFocus
            />
            {emailError ? (
              <p id={`${id}-email-error`} className="text-sm text-destructive">
                {emailError}
              </p>
            ) : null}
          </div>
          <div className="grid gap-2">
            <Label htmlFor={`${id}-password`}>Password</Label>
            <Input
              id={`${id}-password`}
              type="password"
              autoComplete={mode === "login" ? "current-password" : "new-password"}
              value={password}
              maxLength={PASSWORD_MAX_LENGTH}
              onChange={(e) => setPassword(e.target.value)}
              aria-invalid={passwordError ? true : undefined}
              aria-describedby={
                passwordError
                  ? `${id}-password-error`
                  : mode === "signup"
                    ? `${id}-password-hint`
                    : undefined
              }
            />
            {passwordError ? (
              <p id={`${id}-password-error`} className="text-sm text-destructive">
                {passwordError}
              </p>
            ) : mode === "signup" ? (
              <p id={`${id}-password-hint`} className="text-sm text-muted-foreground">
                At least {PASSWORD_MIN_LENGTH} characters.
              </p>
            ) : null}
          </div>
          <Button type="submit" disabled={mutation.isPending || mutation.isSuccess}>
            {mutation.isPending ? <Loader2 className="animate-spin" aria-hidden /> : null}
            {copy.submit}
          </Button>
        </form>
        <p className="mt-6 text-center text-sm text-muted-foreground">
          {mode === "login" ? (
            <>
              New here?{" "}
              <Link
                href={next ? { pathname: "/signup", query: { next } } : "/signup"}
                className="font-medium text-foreground underline underline-offset-4"
              >
                Create an account
              </Link>
            </>
          ) : (
            <>
              Already have an account?{" "}
              <Link
                href={next ? { pathname: "/login", query: { next } } : "/login"}
                className="font-medium text-foreground underline underline-offset-4"
              >
                Sign in
              </Link>
            </>
          )}
        </p>
      </CardContent>
    </Card>
  );
}
