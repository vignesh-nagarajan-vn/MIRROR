/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // NEXT_PUBLIC_API_URL selects where analyze requests go:
  //   - unset  -> same-origin /api/analyze (Vercel serverless, Claude vision)
  //   - a URL  -> that backend (e.g. http://localhost:8000 for local FastAPI)
  // Left unset by default so the hosted (Vercel) build works out of the box.
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || "",
  },
};

module.exports = nextConfig;
