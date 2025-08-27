/**
 * Mermaid initializer for MkDocs Material.
 * - Runs on first load and on each SPA navigation.
 * - Uses loose security to allow inline labels.
 */
(function () {
  function init() {
    if (window.mermaid && typeof window.mermaid.initialize === 'function') {
      window.mermaid.initialize({ startOnLoad: true, securityLevel: 'loose' });
      try { window.mermaid.init(); } catch (e) { /* ignore */ }
    }
  }

  // MkDocs Material emits `document$` on page changes (client-side routing)
  if (window.document$ && typeof window.document$.subscribe === 'function') {
    window.document$.subscribe(init);
  } else {
    document.addEventListener('DOMContentLoaded', init);
  }
})();
