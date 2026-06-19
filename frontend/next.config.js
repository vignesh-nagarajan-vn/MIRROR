/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // The backend base URL is read at runtime from NEXT_PUBLIC_API_URL.
  // Defaults to the local FastAPI server.
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
  },
};

module.exports = nextConfig;
