This project uses bun and uv for package management.

After making changes to server.py, make sure to go to nextjs dir and run `bun run generate-api-client` to update the TypeScript API client.

If writing TypeScript code, run `bunx tsc --noEmit` to check for type errors and fix any that exist. Also if you're in the nextjs dir, run `bun lint` to check for lint errors and fix any that exist.

Assume that I have the Next.js dev server running.
