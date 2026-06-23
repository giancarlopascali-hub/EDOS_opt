// Initialize global state on window for cross-module accessibility
window.currentModule = 'doe';        // Active module: 'doe' | 'bo' | 'sa'
window.currentData = null;           // Full dataset: Array of Arrays (row-major)
window.currentColumns = [];          // Column header names: Array of Strings
window.columnRoles = {};             // Map of column name → 'feature' | 'objective'
window.objectiveConfigs = {};         // Objective settings: {name, type, target, importance, included}
window.boFeatureConfigs = {};         // BO feature settings: {type, range, included}
window.saFeatureConfigs = {};         // SA feature settings: {type, included}
window.suggestionsData = null;       // Latest DoE / BO results from the backend
window.currentDoEFeatures = [];     // Snapshot of DoE feature configs at the time of the last run
window.currentDoEObjectives = [];   // Snapshot of DoE objective configs at the time of the last run
window.currentAbortController = null; // AbortController to cancel in-flight fetch requests
window.lastSAResult = null;         // Cached last SA result for report export
window.saSelectedObjectives = [];   // Cached selected objectives for SA estimation
window.saSelectedFeatures = [];     // Cached selected features for SA estimation
