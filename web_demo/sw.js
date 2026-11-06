const CACHE = 'cone-vision-da3-q4-19';
const ASSETS = [
  './',
  './index.html',
  './styles.css',
  './app.mjs',
  './depth_cloud.mjs',
  './cloud_view.mjs',
  './las.mjs',
  './model_store.mjs',
  './exif_fov.mjs',
  './inference.mjs',
  './runtime_session.mjs',
  './pnp.mjs',
  './manifest.webmanifest',
  './packed_loader.mjs',
  '../.payload/opencv/part-000.txt',
  '../.payload/opencv/part-001.txt',
  '../.payload/opencv/part-002.txt',
  '../.payload/opencv/part-003.txt',
  '../.payload/opencv/part-004.txt',
  './vendor/OpenCV-LICENSE.txt',
  '../assets/cone_geometry.json',
  './icons/icon-192.png',
  './icons/icon-512.png'
];

self.addEventListener('install', event => {
  event.waitUntil(caches.open(CACHE).then(cache => cache.addAll(ASSETS)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', event => {
  event.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(key => key.startsWith('cone-vision-') && key !== CACHE).map(key => caches.delete(key))))
    .then(() => self.clients.claim()));
});

self.addEventListener('fetch', event => {
  if (event.request.method !== 'GET' || /\.(onnx|zip)$/i.test(new URL(event.request.url).pathname)) return;
  event.respondWith(caches.match(event.request).then(hit => hit || fetch(event.request).then(response => {
    const copy = response.clone();
    caches.open(CACHE).then(cache => cache.put(event.request, copy)).catch(() => {});
    return response;
  }).catch(() => hit)));
});
