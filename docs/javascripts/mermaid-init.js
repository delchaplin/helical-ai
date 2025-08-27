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
  // Re-run after in-page navigation (Material)
  if (window.document$ && typeof window.document$.subscribe === 'function') {
    window.document$.subscribe(init);
  } else {
    document.addEventListener('DOMContentLoaded', init);
  }
})();
