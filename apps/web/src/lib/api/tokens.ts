import { api, unwrap, type ApiTokenCreate } from "./client";

export const tokenKeys = {
  list: ["settings", "api-tokens"] as const,
};

export async function fetchApiTokens() {
  return unwrap(await api.GET("/api/v1/settings/api-tokens"), "/api/v1/settings/api-tokens");
}

export async function createApiToken(body: ApiTokenCreate) {
  return unwrap(await api.POST("/api/v1/settings/api-tokens", { body }), "/api/v1/settings/api-tokens");
}

export async function revokeApiToken(tokenId: string) {
  return unwrap(
    await api.DELETE("/api/v1/settings/api-tokens/{token_id}", { params: { path: { token_id: tokenId } } }),
    "/api/v1/settings/api-tokens/{token_id}",
  );
}
