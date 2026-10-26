#!/bin/sh
# Start the Compliance Workspace MCP server over stdio inside the running mcp-server
# container. Claude Desktop on another machine runs it over SSH:
#
#   ssh -T -o BatchMode=yes you@this-host ~/bin/compliance-mcp-stdio
#
# Install on the Linux host that runs the stack (see apps/mcp-server/README.md):
#   install -m 0755 apps/mcp-server/scripts/compliance-mcp-stdio.sh ~/bin/compliance-mcp-stdio
#   install -d -m 0700 ~/.config/compliance-mcp
#   (umask 077; cat > ~/.config/compliance-mcp/token)   # paste the token, then Ctrl-D
#
# The token is read from a file only this user can read and handed to `docker exec` by
# variable *name*, so it never appears in a command line, in the Mac's Desktop config or
# in logs. stdout is the MCP protocol stream: this script prints nothing to it.
set -eu
TOKEN_FILE="${COMPLIANCE_MCP_TOKEN_FILE:-$HOME/.config/compliance-mcp/token}"
CONTAINER="${COMPLIANCE_MCP_CONTAINER:-compliance-mcp-server-1}"
DOCKER="${DOCKER:-docker}"

fail() { echo "compliance-mcp-stdio: $*" >&2; exit 2; }

[ -r "$TOKEN_FILE" ] || fail "cannot read the token file $TOKEN_FILE"
mode=$(stat -c %a "$TOKEN_FILE" 2>/dev/null || stat -f %Lp "$TOKEN_FILE")
case "$mode" in
  400 | 600) ;;
  *) fail "$TOKEN_FILE must be readable by you only (chmod 600), not $mode" ;;
esac
COMPLIANCE_MCP_TOKEN=$(cat "$TOKEN_FILE")
[ -n "$COMPLIANCE_MCP_TOKEN" ] || fail "$TOKEN_FILE is empty"
export COMPLIANCE_MCP_TOKEN
exec "$DOCKER" exec -i -e COMPLIANCE_MCP_TOKEN "$CONTAINER" \
  python -m compliance_mcp_server --transport stdio
