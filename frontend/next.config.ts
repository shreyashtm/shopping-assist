import path from "node:path";

import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /*
   * Pinned explicitly: an unrelated package-lock.json in the user's home
   * directory otherwise makes Turbopack infer the workspace root as ~, which
   * would pull the entire home directory into module resolution.
   */
  turbopack: { root: path.resolve(".") },
  images: {
    /*
     * Catalogue images are hotlinked from retailer CDNs, so each host must be
     * allowlisted. Images render `unoptimized` (see ProductCard): passing
     * third-party retailer images through our own optimizer would proxy their
     * bandwidth through our server for no gain, since they are already
     * CDN-served at display size.
     */
    remotePatterns: [
      { protocol: "https", hostname: "picsum.photos" },
      { protocol: "https", hostname: "**.decathlon.in" },
      { protocol: "https", hostname: "**.media-amazon.com" },
      { protocol: "https", hostname: "**.myntassets.com" },
      { protocol: "https", hostname: "**.nykaa.com" },
    ],
  },
};

export default nextConfig;
