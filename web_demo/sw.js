const CACHE = 'cone-vision-v2-exif';
const ASSETS = [
  './',
  './index.html',
  './styles.css',
  './app.mjs',
  './exif_fov.mjs',
  './inference.mjs',
  './pnp.mjs',
  './manifest.webmanifest',
  './packed_loader.mjs',
  '../.payload/detector/part-000.txt',
  '../.payload/detector/part-001.txt',
  '../.payload/detector/part-002.txt',
  '../.payload/detector/part-003.txt',
  '../.payload/detector/part-004.txt',
  '../.payload/opencv/part-000.txt',
  '../.payload/opencv/part-001.txt',
  '../.payload/opencv/part-002.txt',
  '../.payload/opencv/part-003.txt',
  '../.payload/opencv/part-004.txt',
  './vendor/OpenCV-LICENSE.txt',
  '../assets/cone_geometry.json',
  '../models/retrain_v7/cone_keypoints.onnx',
  '../models/retrain_v7/cone_detector.interface.json',
  '../models/retrain_v7/cone_keypoints.interface.json',
  './icons/icon-192.png',
  './icons/icon-512.png'
];

self.addEventListener('install', event => {
  event.waitUntil(caches.open(CACHE).then(cache => cache.addAll(ASSETS)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', event => {
  event.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(key => key !== CACHE).map(key => caches.delete(key))))
    .then(() => self.clients.claim()));
});

self.addEventListener('fetch', event => {
  if (event.request.method !== 'GET') return;
  event.respondWith(caches.match(event.request).then(hit => hit || fetch(event.request).then(response => {
    const copy = response.clone();
    caches.open(CACHE).then(cache => cache.put(event.request, copy)).catch(() => {});
    return response;
  }).catch(() => hit)));
});
