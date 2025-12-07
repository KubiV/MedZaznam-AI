const CACHE_NAME = 'med-recorder-v1';
const ASSETS = [
  '/static/web-app-manifest-192x192.png',
  '/static/web-app-manifest-512x512.png',
  '/static/favicon.ico',
  'https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css',
  'https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css'
];

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(ASSETS))
  );
});

self.addEventListener('fetch', (e) => {
  // Pro Socket.IO a API požadavky nepoužíváme cache
  if (e.request.method !== 'GET' || e.request.url.includes('socket.io')) {
    return;
  }
  
  e.respondWith(
    fetch(e.request).catch(() => caches.match(e.request))
  );
});