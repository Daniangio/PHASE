import React, { useState, useEffect, useRef } from 'react';
import { 
  Upload, Activity, Server, CheckCircle, XCircle, AlertTriangle, 
  Loader2, Database, FileText, ChevronRight, ArrowLeft, Brain, Sliders, Zap,
  Eye, Save, Trash2, FolderDown, Download,
  Palette,
  Target // Add Target icon for QUBO
} from 'lucide-react';

/*
================================================================================
Main Application Component (App)
================================================================================
*/
// ... (Main App component remains unchanged) ...
export default function App() {
  const [page, setPage] = useState('submit');
  const [pollingJobId, setPollingJobId] = useState(null); // This is the RQ_JOB_ID
  const [selectedResultId, setSelectedResultId] = useState(null); // This is the JOB_UUID

  // State for the submission form, lifted up to preserve it across pages
  const [formState, setFormState] = useState({
    analysisType: 'static',
    files: {
      active_topo: null, active_traj: null,
      inactive_topo: null, inactive_traj: null,
      config: null,
    },
    teLag: 10,
    targetSelection: 'resid 131',
    activeSlice: '',
    inactiveSlice: '',
    selectionMode: 'all',
    manualSelections: 'resid 50\nresid 131',
    quboLambda: 1.0,
    quboSolutions: 5,
    quboCvFolds: 3,
    quboNEstimators: 50,
  });

  // State for user-defined custom selections, persisted in localStorage
  const [customSelections, setCustomSelections] = useState(() => {
    const saved = localStorage.getItem('alloskin_custom_selections');
    return saved ? JSON.parse(saved) : [];
  });

  const navigateToStatus = (rqJobId) => {
    setPollingJobId(rqJobId);
    setPage('status');
  };

  const navigateToResults = () => {
    setPage('results');
  };

  const navigateToResultDetail = (jobUUID) => {
    setSelectedResultId(jobUUID);
    setPage('result_detail');
  };

  const navigateToVisualize = (jobUUID) => {
    setSelectedResultId(jobUUID);
    setPage('visualize_result');
  };

  const renderPage = () => {
    switch (page) {
      case 'submit':
        return <SubmitJobPage 
                 formState={formState}
                 setFormState={setFormState}
                 customSelections={customSelections}
                 setCustomSelections={setCustomSelections}
                 onJobSubmitted={navigateToStatus} />;
      case 'status':
        return (
          <JobStatusPage 
            jobId={pollingJobId} 
            onNavigateToResults={navigateToResults}
            onNavigateToResultDetail={navigateToResultDetail}
          />
        );
      case 'health':
        return <HealthCheckPage />;
      case 'results':
        return (
          <ResultsListPage 
            onSelectResult={navigateToResultDetail}
            onSelectRunningJob={navigateToStatus}
          />
        );
      case 'result_detail':
        return (
          <ResultDetailPage 
            resultId={selectedResultId} 
            onBack={() => setPage('results')} 
            onVisualize={navigateToVisualize}
          />
        );
      case 'visualize_result':
        return (
          <VisualizeResultPage 
            resultId={selectedResultId} 
            onBack={() => setPage('result_detail')} 
          />
        );
      default:
        return <SubmitJobPage 
                formState={formState}
                setFormState={setFormState}
                customSelections={customSelections}
                setCustomSelections={setCustomSelections}
                onJobSubmitted={navigateToStatus} />;
    }
  };

  return (
    <div className="flex flex-col min-h-screen bg-gray-900 text-gray-100 font-inter">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 shadow-lg">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <div className="flex items-center space-x-3">
            <Zap className="h-8 w-8 text-cyan-400" />
            <h1 className="text-2xl font-bold tracking-tight text-white">AllosKin</h1>
          </div>
          <Navbar setPage={setPage} currentPage={page} />
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-grow container mx-auto px-4 py-8">
        {renderPage()}
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 text-gray-400 text-sm text-center py-4 border-t border-gray-700">
        AllosKin Causal Analysis Pipeline
      </footer>
    </div>
  );
}

/*
================================================================================
Navigation Component (Navbar)
================================================================================
*/
// ... (Navbar component remains unchanged) ...
const Navbar = ({ setPage, currentPage }) => {
  const navItems = [
    { name: 'submit', label: 'Submit Job', icon: <Upload className="h-4 w-4" /> },
    { name: 'results', label: 'View Results', icon: <Database className="h-4 w-4" /> },
    { name: 'health', label: 'System Health', icon: <Server className="h-4 w-4" /> },
  ];

  const getLinkClasses = (pageName) => {
    const isResultsActive = ['results', 'status', 'result_detail', 'visualize_result'].includes(currentPage);
    
    let isActive = currentPage === pageName;
    if (pageName === 'results' && isResultsActive) {
      isActive = true;
    }
    if (pageName === 'submit' && isResultsActive) {
      isActive = false;
    }

    return `flex items-center space-x-2 px-3 py-2 rounded-md font-medium transition-colors ${
      isActive
        ? 'bg-cyan-600 text-white'
        : 'text-gray-300 hover:bg-gray-700 hover:text-white'
    }`;
  };

  return (
    <nav className="flex space-x-2">
      {navItems.map((item) => (
        <button
          key={item.name}
          onClick={() => setPage(item.name)}
          className={getLinkClasses(item.name)}
        >
          {item.icon}
          <span>{item.label}</span>
        </button>
      ))}
      {(currentPage === 'status' && !['results', 'result_detail', 'visualize_result'].includes(currentPage)) && (
        <button
          onClick={() => setPage('status')}
          className={getLinkClasses('status')}
        >
          <Loader2 className="h-4 w-4 animate-spin" />
          <span>Job Status</span>
        </button>
      )}
    </nav>
  );
};


/*
================================================================================
Shared Utility Components (Errors, Loaders, Upload)
================================================================================
*/
// ... (ErrorDisplay, FullPageLoader, FileDropzone, xhrUpload remain unchanged) ...
const ErrorDisplay = ({ error }) => (
  <div className="bg-red-900 border border-red-700 text-red-100 p-3 rounded-md flex items-center space-x-2">
    <XCircle className="h-5 w-5" /> <span>{error}</span>
  </div>
);

const FullPageLoader = () => (
    <div className="flex justify-center items-center h-64">
        <Loader2 className="h-12 w-12 text-cyan-400 animate-spin" />
    </div>
);

const FileDropzone = ({ name, label, file, onChange }) => {
  const [isDragging, setIsDragging] = useState(false);
  const dragCounter = useRef(0);

  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current++;
    if (!isDragging) {
      setIsDragging(true);
    }
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current--;
    if (dragCounter.current === 0) {
      setIsDragging(false);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    dragCounter.current = 0;

    const droppedFiles = e.dataTransfer.files;
    if (droppedFiles && droppedFiles.length > 0) {
      const syntheticEvent = {
        target: {
          name: name,
          files: droppedFiles,
        },
      };
      onChange(syntheticEvent);
    }
  };

  return (
    <div>
      <label className="block text-sm font-medium text-gray-300 mb-1">{label}</label>
      <div
        className={`mt-1 flex justify-center px-6 pt-5 pb-6 border-2 ${
          isDragging ? 'border-cyan-500' : 'border-gray-600'
        } border-dashed rounded-md transition-colors`}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      >
        <div className="space-y-1 text-center">
          <svg className="mx-auto h-10 w-10 text-gray-500" stroke="currentColor" fill="none" viewBox="0 0 48 48" aria-hidden="true">
            <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <div className="flex text-sm text-gray-400">
            <label htmlFor={name} className="relative cursor-pointer bg-gray-900 rounded-md font-medium text-cyan-500 hover:text-cyan-400 focus-within:outline-none focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-offset-gray-900 focus-within:ring-cyan-500">
              <span>Upload a file</span>
              <input id={name} name={name} type="file" className="sr-only" onChange={onChange} />
            </label>
            <p className="pl-1">or drag and drop</p>
          </div>
          {file ? (
            <p className="text-xs text-green-400 truncate max-w-xs">{file.name}</p>
          ) : (
            <p className="text-xs text-gray-500">Up to 500MB</p>
          )}
        </div>
      </div>
    </div>
  );
};

function xhrUpload(url, formData, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    
    xhr.open("POST", url, true);

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        const percentComplete = Math.round((e.loaded / e.total) * 100);
        onProgress(percentComplete);
      }
    };

    xhr.onload = () => {
      let responseData;
      try {
        if (xhr.responseText) {
          responseData = JSON.parse(xhr.responseText);
        } else {
          responseData = {};
        }
      } catch (err) {
        reject(new Error(`Failed to parse server response: ${xhr.responseText}`));
        return;
      }

      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(responseData);
      } else {
        const errorDetail = responseData.detail || "Job submission failed.";
        const errorMsg = typeof errorDetail === 'object' ? JSON.stringify(errorDetail) : errorDetail;
        reject(new Error(errorMsg));
      }
    };

    xhr.onerror = () => {
      reject(new Error("Network error occurred during upload."));
    };

    xhr.onabort = () => {
      reject(new Error("Upload was cancelled."));
    };

    xhr.send(formData);
  });
}


/*
================================================================================
Page: Submit Job (SubmitJobPage)
================================================================================
*/
const SubmitJobPage = ({ formState, setFormState, customSelections, setCustomSelections, onJobSubmitted }) => {
  // Local state for UI feedback (loading, errors)
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);

  // Destructure state and create setters for easier use
  const {
    analysisType, files, teLag, targetSelection, activeSlice,
    inactiveSlice, selectionMode, manualSelections, quboLambda,
    quboSolutions, quboCvFolds, quboNEstimators
  } = formState;

  const setAnalysisType = (value) => setFormState(prev => ({ ...prev, analysisType: value }));
  const setFiles = (updater) => setFormState(prev => ({ ...prev, files: typeof updater === 'function' ? updater(prev.files) : updater }));
  const setTeLag = (value) => setFormState(prev => ({ ...prev, teLag: value }));
  const setTargetSelection = (value) => setFormState(prev => ({ ...prev, targetSelection: value }));
  const setActiveSlice = (value) => setFormState(prev => ({ ...prev, activeSlice: value }));
  const setInactiveSlice = (value) => setFormState(prev => ({ ...prev, inactiveSlice: value }));
  const setSelectionMode = (value) => setFormState(prev => ({ ...prev, selectionMode: value }));
  const setManualSelections = (value) => setFormState(prev => ({ ...prev, manualSelections: value }));
  const setQuboLambda = (value) => setFormState(prev => ({ ...prev, quboLambda: value }));
  const setQuboSolutions = (value) => setFormState(prev => ({ ...prev, quboSolutions: value }));
  const setQuboCvFolds = (value) => setFormState(prev => ({ ...prev, quboCvFolds: value }));
  const setQuboNEstimators = (value) => setFormState(prev => ({ ...prev, quboNEstimators: value }));

  // Persist custom selections to localStorage whenever they change
  useEffect(() => {
    localStorage.setItem('alloskin_custom_selections', JSON.stringify(customSelections));
  }, [customSelections]);

  const handleFileChange = (e) => {
    const { name, files: selectedFiles } = e.target;
    setFiles(prev => ({ ...prev, [name]: selectedFiles[0] }));
  };

  const handleSliceChange = (e) => {
    const { name, value } = e.target;
    if (name === 'active_slice') setActiveSlice(value);
    if (name === 'inactive_slice') setInactiveSlice(value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    setUploadProgress(0);

    const requiredFiles = ['active_topo', 'active_traj', 'inactive_topo', 'inactive_traj'];
    if (requiredFiles.some(key => !files[key])) {
      setError("Please upload all 4 trajectory and topology files.");
      setIsLoading(false);
      return;
    }

    if (selectionMode === 'file' && !files.config) {
      setError("Please upload a config file or change the selection method.");
      setIsLoading(false);
      return;
    }

    const formData = new FormData();
    formData.append('active_topo', files.active_topo);
    formData.append('active_traj', files.active_traj);
    formData.append('inactive_topo', files.inactive_topo);
    formData.append('inactive_traj', files.inactive_traj);

    if (selectionMode === 'file' && files.config) {
      formData.append('config', files.config);
    } else if (selectionMode === 'manual' && manualSelections.trim()) {
      try {
        const selectionsDict = manualSelections
          .split('\n')
          .map(line => line.trim())
          .filter(line => line)
          .reduce((acc, line) => {
            const key = line.replace(/[^a-zA-Z0-9]/g, '_');
            acc[key] = line;
            return acc;
          }, {});
        formData.append('residue_selections_json', JSON.stringify(selectionsDict));
      } catch (err) {
        setError("Failed to parse manual selections. Please check the format.");
        setIsLoading(false);
        return;
      }
    }

    let endpoint = '';
    switch (analysisType) {
      case 'static':
        endpoint = '/api/v1/submit/static';
        break;
      case 'qubo':
        endpoint = '/api/v1/submit/qubo';
        if (!targetSelection) {
            setError("Target Selection string is required for QUBO.");
            setIsLoading(false);
            return;
        }
        formData.append('target_selection_string', targetSelection);
        formData.append('lambda_redundancy', quboLambda);
        formData.append('num_solutions', quboSolutions);
        formData.append('qubo_cv_folds', quboCvFolds);
        formData.append('qubo_n_estimators', quboNEstimators);
        break;
      case 'dynamic':
        endpoint = '/api/v1/submit/dynamic';
        formData.append('te_lag', teLag);
        break;
      default:
        setError("Invalid analysis type selected.");
        setIsLoading(false);
        return;
    }

    if (activeSlice) formData.append('active_slice', activeSlice);
    if (inactiveSlice) formData.append('inactive_slice', inactiveSlice);

    try {
      const data = await xhrUpload(endpoint, formData, setUploadProgress);
      
      if (!data.job_id) {
          throw new Error("Submission succeeded but did not return a job ID.");
      }
      onJobSubmitted(data.job_id);
    } catch (err) {
      setError(err.message || "Upload failed. Please check the console.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto">
      <h2 className="text-3xl font-bold mb-6 text-white">Run New Analysis</h2>
      <div className="flex space-x-1 rounded-lg bg-gray-800 p-1 mb-6">
        <TabButton 
          icon={<Brain />} label="Static Reporters" 
          isActive={analysisType === 'static'} 
          onClick={() => setAnalysisType('static')} 
        />
        <TabButton 
          icon={<Sliders />} label="QUBO Optimal Set" 
          isActive={analysisType === 'qubo'} 
          onClick={() => setAnalysisType('qubo')} 
        />
        <TabButton 
          icon={<Zap />} label="Dynamic TE" 
          isActive={analysisType === 'dynamic'} 
          onClick={() => setAnalysisType('dynamic')} 
        />
      </div>
      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <FileInputGroup
            title="Active State"
            files={[
              { name: 'active_topo', label: 'Topology (PDB, GRO, ...)' },
              { name: 'active_traj', label: 'Trajectory (XTC, TRR, ...)' },
            ]}
            sliceName="active_slice"
            sliceValue={activeSlice}
            onSliceChange={handleSliceChange}
            fileState={files}
            onChange={handleFileChange}
          />
          <FileInputGroup
            title="Inactive State"
            files={[
              { name: 'inactive_topo', label: 'Topology (PDB, GRO, ...)' },
              { name: 'inactive_traj', label: 'Trajectory (XTC, TRR, ...)' },
            ]}
            sliceName="inactive_slice"
            sliceValue={inactiveSlice}
            onSliceChange={handleSliceChange}
            fileState={files}
            onChange={handleFileChange}
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
            <h3 className="text-lg font-semibold mb-3 text-cyan-400">Candidate Residues (Input Set)</h3>
            <div className="flex space-x-1 rounded-lg bg-gray-900 p-1 mb-4">
              <TabButton label="Analyze All" isActive={selectionMode === 'all'} onClick={() => setSelectionMode('all')} />
              <TabButton label="Upload File" isActive={selectionMode === 'file'} onClick={() => setSelectionMode('file')} />
              <TabButton label="Enter Manually" isActive={selectionMode === 'manual'} onClick={() => setSelectionMode('manual')} />
            </div>
            {selectionMode === 'all' && (
              <div className="text-sm text-gray-400 p-4 bg-gray-900 rounded-md">
                All common protein residues will be used as the candidate pool for analysis.
              </div>
            )}
            {selectionMode === 'file' && (
              <FileDropzone
                name="config" label="Residue Config (config.yml)"
                file={files.config} onChange={handleFileChange}
              />
            )}
            {selectionMode === 'manual' && (
              <div>
                <SelectionField
                  label="Enter MDAnalysis Selections (one per line)"
                  value={manualSelections}
                  onChange={setManualSelections}
                  selectionType="manual"
                  customSelections={customSelections}
                  setCustomSelections={setCustomSelections}
                />
                <p className="text-xs text-gray-500 mt-1">Each line will become a feature. Keys are auto-generated.</p>
              </div>
            )}
          </div>
          <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
            <h3 className="text-lg font-semibold mb-3 text-cyan-400">Analysis Parameters</h3>
            {analysisType === 'static' && (
              <p className="text-sm text-gray-400">No additional parameters required for Static analysis.</p>
            )}
            {analysisType === 'qubo' && (
              <div className="space-y-4">
                <SelectionField
                  label="Target Selection (MDAnalysis string)"
                  value={targetSelection}
                  onChange={setTargetSelection}
                  selectionType="target"
                  customSelections={customSelections}
                  setCustomSelections={setCustomSelections}
                />
                <p className="text-xs text-gray-500 mt-1">
                  The residue(s) to predict. These will be removed from the candidate pool.
                </p>
                <div className="grid grid-cols-2 gap-4 pt-2">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-1">Lambda Penalty</label>
                    <input type="number" step="0.1" value={quboLambda} onChange={(e) => setQuboLambda(e.target.value)}
                           className="w-full bg-gray-900 border border-gray-600 rounded-md p-2 text-white focus:ring-cyan-500 focus:border-cyan-500" />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-1"># Solutions</label>
                    <input type="number" value={quboSolutions} onChange={(e) => setQuboSolutions(e.target.value)}
                           className="w-full bg-gray-900 border border-gray-600 rounded-md p-2 text-white focus:ring-cyan-500 focus:border-cyan-500" />
                  </div>
                </div>
                <details className="text-sm text-gray-400 mt-2">
                  <summary className="cursor-pointer hover:text-white">Advanced RF Parameters</summary>
                  <div className="grid grid-cols-2 gap-4 pt-2 mt-2 border-t border-gray-700">
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">CV Folds</label>
                      <input
                        type="number"
                        value={quboCvFolds}
                        onChange={(e) => setQuboCvFolds(e.target.value)}
                        className="w-full bg-gray-900 border border-gray-600 rounded-md p-2 text-white focus:ring-cyan-500 focus:border-cyan-500"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">N Estimators</label>
                      <input
                        type="number"
                        value={quboNEstimators}
                        onChange={(e) => setQuboNEstimators(e.target.value)}
                        className="w-full bg-gray-900 border border-gray-600 rounded-md p-2 text-white focus:ring-cyan-500 focus:border-cyan-500"
                      />
                    </div>
                  </div>
                </details>
              </div>
            )}
            {analysisType === 'dynamic' && (
              <div>
                <label htmlFor="te_lag" className="block text-sm font-medium text-gray-300 mb-1">
                  TE Lag Time (frames)
                </label>
                <input
                  type="number" id="te_lag" value={teLag}
                  onChange={(e) => setTeLag(e.target.value)}
                  className="w-full bg-gray-900 border border-gray-600 rounded-md p-2 text-white focus:ring-cyan-500 focus:border-cyan-500"
                />
              </div>
            )}
          </div>
        </div>
        {error && <ErrorDisplay error={error} />}
        
        {isLoading && (
          <UploadProgress progress={uploadProgress} />
        )}
        
        <SubmitButton isLoading={isLoading} />
      </form>
    </div>
  );
};

const SelectionField = ({ label, value, onChange, selectionType, customSelections, setCustomSelections }) => {
  const [showLoadPopup, setShowLoadPopup] = useState(false);
  const wrapperRef = useRef(null);

  // Close popup when clicking outside
  useEffect(() => {
    function handleClickOutside(event) {
      if (wrapperRef.current && !wrapperRef.current.contains(event.target)) {
        setShowLoadPopup(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [wrapperRef]);

  const handleSave = () => {
    if (!value.trim()) {
      alert('There is no content to save.');
      return;
    }
    const name = window.prompt('Enter a name for this selection:');
    if (!name || !name.trim()) {
      return;
    }
    const newSelection = { id: Date.now(), name: name.trim(), type: selectionType, content: value };
    setCustomSelections(prev => [...prev, newSelection]);
  };

  const handleLoad = (content) => {
    onChange(content);
    setShowLoadPopup(false);
  };

  const deleteSelection = (id) => {
    setCustomSelections(prev => prev.filter(s => s.id !== id));
  };

  const relevantSelections = customSelections.filter(s => s.type === selectionType);

  const InputComponent = selectionType === 'manual' ? 'textarea' : 'input';

  return (
    <div className="relative" ref={wrapperRef}>
      <label className="block text-sm font-medium text-gray-300 mb-1">{label}</label>
      <div className="flex items-center space-x-2">
        <InputComponent
          rows={selectionType === 'manual' ? 4 : undefined}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="w-full bg-gray-900 border border-gray-600 rounded-md p-2 text-white focus:ring-cyan-500 focus:border-cyan-500 font-mono text-sm"
          placeholder={selectionType === 'manual' ? "e.g., resid 50\nresid 130 to 150" : "e.g., resid 131"}
        />
        <div className="flex flex-col space-y-2">
          <button type="button" onClick={handleSave} title="Save current selection" className="p-2 bg-gray-700 hover:bg-gray-600 rounded-md text-white transition-colors">
            <Save className="h-5 w-5" />
          </button>
          <button type="button" onClick={() => setShowLoadPopup(p => !p)} disabled={relevantSelections.length === 0} title="Load a saved selection" className="p-2 bg-gray-700 hover:bg-gray-600 rounded-md text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed">
            <FolderDown className="h-5 w-5" />
          </button>
        </div>
      </div>
      {showLoadPopup && relevantSelections.length > 0 && (
        <LoadSelectionsPopup
          selections={relevantSelections}
          onLoad={handleLoad}
          onDelete={deleteSelection}
          onClose={() => setShowLoadPopup(false)}
        />
      )}
    </div>
  );
};

const LoadSelectionsPopup = ({ selections, onLoad, onDelete, onClose }) => {
  return (
    <div className="absolute top-full right-0 mt-2 w-72 bg-gray-800 border border-gray-600 rounded-lg shadow-lg z-10">
      <div className="flex justify-between items-center p-2 border-b border-gray-700">
        <h4 className="font-semibold text-white text-sm">Load Selection</h4>
        <button onClick={onClose} className="text-gray-400 hover:text-white">
          <XCircle className="h-5 w-5" />
        </button>
      </div>
      <ul className="max-h-60 overflow-y-auto p-1">
        {selections.map(sel => (
          <li key={sel.id} className="group flex items-center justify-between p-2 rounded-md hover:bg-gray-700">
            <button
              type="button"
              onClick={() => onLoad(sel.content)}
              className="text-left w-full"
            >
              <p className="font-medium text-white truncate">{sel.name}</p>
              <p className="text-xs text-gray-400 truncate">
                {sel.content.replace(/\n/g, '; ')}
              </p>
            </button>
            <button
              type="button"
              onClick={() => onDelete(sel.id)}
              className="text-gray-500 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-opacity ml-2 p-1"
              title="Delete selection"
            >
              <Trash2 className="h-4 w-4" />
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
};

const TabButton = ({ icon, label, isActive, onClick }) => (
  <button
    type="button" onClick={onClick}
    className={`w-full flex justify-center items-center space-x-2 px-3 py-3 font-medium text-sm rounded-md transition-colors ${
      isActive ? 'bg-cyan-600 text-white' : 'text-gray-300 hover:bg-gray-700 hover:text-white'
    }`}
  >
    {icon} <span>{label}</span>
  </button>
);

const FileInputGroup = ({ title, files, fileState, onChange, sliceName, sliceValue, onSliceChange }) => (
  <div className="bg-gray-800 p-4 rounded-lg border border-gray-700 space-y-4">
    <h3 className="text-lg font-semibold text-cyan-400">{title}</h3>
    {files.map((file) => (
      <FileDropzone
        key={file.name} name={file.name} label={file.label}
        file={fileState[file.name]} onChange={onChange}
      />
    ))}
    <SliceInput name={sliceName} value={sliceValue} onChange={onSliceChange} />
  </div>
);

const SliceInput = ({ name, value, onChange }) => (
  <div>
    <label htmlFor={name} className="block text-sm font-medium text-gray-300 mb-1">
      Trajectory Slice (Optional)
    </label>
    <input
      type="text" id={name} name={name} value={value} onChange={onChange}
      className="w-full bg-gray-900 border border-gray-600 rounded-md p-2 text-white focus:ring-cyan-500 focus:border-cyan-500 font-mono text-sm"
      placeholder="start:stop:step"
    />
  </div>
);

const SubmitButton = ({ isLoading }) => (
  <button
    type="submit" disabled={isLoading}
    className="w-full flex justify-center items-center space-x-2 bg-cyan-600 hover:bg-cyan-700 text-white font-bold py-3 px-6 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
  >
    {isLoading ? <Loader2 className="h-6 w-6 animate-spin" /> : <Upload className="h-6 w-6" />}
    <span>{isLoading ? 'Uploading...' : 'Submit Job'}</span>
  </button>
);

const UploadProgress = ({ progress }) => (
  <div className="bg-gray-800 p-4 rounded-lg border border-gray-700 space-y-2">
    <p className="text-sm font-medium text-gray-300">Uploading files...</p>
    <div className="w-full bg-gray-700 rounded-full h-2.5">
      <div 
        className="bg-cyan-500 h-2.5 rounded-full transition-all duration-300" 
        style={{ width: `${progress}%` }}
      ></div>
    </div>
    <p className="text-center text-cyan-400 font-mono text-lg">{progress}%</p>
  </div>
);


/*
================================================================================
Page: Job Status (JobStatusPage)
================================================================================
*/
// ... (JobStatusPage and its sub-components remain unchanged) ...
const JobStatusPage = ({ jobId, onNavigateToResults, onNavigateToResultDetail }) => {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const pollingInterval = useRef(null);

  useEffect(() => {
    if (!jobId) {
      setError("No Job ID specified. Redirecting to Results.");
      const timer = setTimeout(() => onNavigateToResults(), 2000);
      return () => clearTimeout(timer);
    }
    const pollStatus = async () => {
      try {
        const response = await fetch(`/api/v1/job/status/${jobId}`);
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Failed to fetch job status.");
        setStatus(data);
        if (data.status === 'finished' || data.status === 'failed') {
          clearInterval(pollingInterval.current);
        }
      } catch (err) {
        setError(err.message);
        clearInterval(pollingInterval.current);
      }
    };
    pollStatus();
    pollingInterval.current = setInterval(pollStatus, 3000);
    return () => clearInterval(pollingInterval.current);
  }, [jobId, onNavigateToResults]);

  if (error) {
    return (
      <div className="max-w-2xl mx-auto text-center">
        <h2 className="text-3xl font-bold mb-4 text-red-500">Error</h2>
        <p className="text-gray-300">{error}</p>
      </div>
    );
  }

  const jobStatus = status?.status || 'queued';
  const metaStatus = status?.meta?.status || (jobStatus === 'queued' ? 'Waiting in queue...' : 'Initializing...');
  const progress = status?.meta?.progress || (jobStatus === 'queued' ? 0 : 5);
  const resultPayload = status?.result;

  if (jobStatus === 'finished' && resultPayload) {
    return (
      <StatusDisplay
        icon={<CheckCircle className="h-16 w-16 text-green-500" />}
        title="Analysis Complete" jobId={jobId} message="Your results are ready."
      >
        <div className="mt-6 text-left">
          <h3 className="text-lg font-semibold text-green-400 mb-2">Results Summary:</h3>
          <pre className="bg-gray-900 p-4 rounded-md text-gray-200 text-xs overflow-auto">
            {JSON.stringify(resultPayload.results || resultPayload, null, 2)}
          </pre>
        </div>
        <button
          onClick={() => onNavigateToResultDetail(resultPayload.job_id)}
          className="mt-6 w-full flex justify-center items-center space-x-2 bg-cyan-600 hover:bg-cyan-700 text-white font-bold py-3 px-6 rounded-lg transition-colors"
        >
          <span>View Result Details</span>
        </button>
      </StatusDisplay>
    );
  }

  if (jobStatus === 'failed') {
    const errorMsg = resultPayload?.error || "An unknown error occurred.";
    return (
      <StatusDisplay
        icon={<XCircle className="h-16 w-16 text-red-500" />}
        title="Analysis Failed" jobId={jobId} message="An error occurred during processing."
      >
        <div className="mt-6 text-left">
          <h3 className="text-lg font-semibold text-red-400 mb-2">Error Details:</h3>
          <pre className="bg-gray-900 p-4 rounded-md text-red-300 text-xs overflow-auto">
            {errorMsg}
          </pre>
        </div>
      </StatusDisplay>
    );
  }

  return (
    <StatusDisplay
      icon={<Loader2 className="h-16 w-16 text-cyan-400 animate-spin" />}
      title="Analysis in Progress..." jobId={jobId} message={metaStatus}
    >
      <div className="w-full bg-gray-700 rounded-full h-2.5 mt-6">
        <div 
          className="bg-cyan-500 h-2.5 rounded-full transition-all" 
          style={{ width: `${progress || 0}%` }}
        ></div>
      </div>
    </StatusDisplay>
  );
};

const StatusDisplay = ({ icon, title, message, jobId, children }) => (
  <div className="max-w-3xl mx-auto bg-gray-800 rounded-lg border border-gray-700 shadow-xl p-8 text-center">
    <div className="flex justify-center mb-6">{icon}</div>
    <h2 className="text-3xl font-bold mb-3 text-white">{title}</h2>
    <p className="text-gray-300 mb-6">{message}</p>
    <div className="bg-gray-900 p-3 rounded-md text-sm text-gray-400 font-mono">
      Polling Job ID: {jobId}
    </div>
    {children}
  </div>
);


/*
================================================================================
Page: Results List (ResultsListPage)
================================================================================
*/
const ResultsListPage = ({ onSelectResult, onSelectRunningJob }) => {
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchResults = async () => {
      setIsLoading(true); setError(null);
      try {
        const response = await fetch('/api/v1/results');
        if (!response.ok) {
          const data = await response.json();
          throw new Error(data.detail || "Failed to fetch results.");
        }
        const data = await response.json();
        setResults(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setIsLoading(false);
      }
    };
    fetchResults();
  }, []);

  const handleDeleteResult = (deletedJobId) => {
    setResults(prevResults => prevResults.filter(r => r.job_id !== deletedJobId));
  };

  const groupResults = (results) => {
    if (!results) return {};
    return results.reduce((acc, result) => {
      const type = result.analysis_type || 'unknown';
      if (!acc[type]) acc[type] = [];
      acc[type].push(result);
      return acc;
    }, {});
  };

  if (isLoading) return <FullPageLoader />;
  if (error) return <ErrorDisplay error={error} />;
  
  const groupedResults = groupResults(results);

  return (
    <div className="max-w-4xl mx-auto">
      <h2 className="text-3xl font-bold mb-6 text-white">Analysis Results</h2>
      {Object.keys(groupedResults).length === 0 && (
        <div className="text-center text-gray-400 bg-gray-800 p-8 rounded-lg border border-gray-700">
          <FileText className="h-12 w-12 mx-auto mb-4 text-gray-500" />
          <h3 className="text-xl font-semibold text-white">No Results Found</h3>
          <p className="mt-2">Run a new job from the "Submit Job" page to see results here.</p>
        </div>
      )}
      <div className="space-y-8">
        {Object.entries(groupedResults).map(([type, items]) => (
          <div key={type}>
            <h3 className="text-2xl font-semibold text-cyan-400 mb-4 capitalize">{type} Analysis</h3>
            <div className="bg-gray-800 rounded-lg border border-gray-700 shadow-lg">
              <ul className="divide-y divide-gray-700">
                {items.map((item) => (
                  <ResultItem 
                    key={item.job_id} 
                    item={item} 
                    onSelectResult={onSelectResult}
                    onSelectRunningJob={onSelectRunningJob}
                    onDelete={handleDeleteResult}
                  />
                ))}
              </ul>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

const ResultItem = ({ item, onSelectResult, onSelectRunningJob, onDelete }) => {
  const status = item.status || 'unknown';
  
  let icon, text, date, handler, classes;
  
  const formattedDate = (dateStr, prefix = "") => {
    if (!dateStr) return "No date";
    return `${prefix}${new Date(dateStr).toLocaleString()}`;
  };

  const handleDeleteClick = async (e) => {
    e.stopPropagation(); // Prevent navigating when clicking the delete button

    if (window.confirm(`Are you sure you want to permanently delete job ${item.job_id}? This action cannot be undone.`)) {
      try {
        const response = await fetch(`/api/v1/results/${item.job_id}`, {
          method: 'DELETE',
        });

        if (!response.ok) {
          const data = await response.json();
          throw new Error(data.detail || 'Failed to delete job.');
        }
        // Notify parent to remove this item from the list
        onDelete(item.job_id);
      } catch (err) {
        alert(`Error: ${err.message}`);
      }
    }
  };
  switch (status) {
    case 'finished':
      icon = <CheckCircle className="h-5 w-5 text-green-500" />;
      text = item.job_id;
      date = formattedDate(item.completed_at, "Completed ");
      handler = () => onSelectResult(item.job_id);
      classes = "hover:bg-gray-700 cursor-pointer";
      break;
    case 'started':
    case 'queued':
      icon = <Loader2 className="h-5 w-5 text-cyan-400 animate-spin" />;
      text = item.job_id;
      date = formattedDate(item.created_at, "Started ");
      handler = () => onSelectRunningJob(item.rq_job_id);
      classes = "hover:bg-gray-700 cursor-pointer";
      break;
    case 'failed':
      icon = <XCircle className="h-5 w-5 text-red-500" />;
      text = item.job_id;
      date = formattedDate(item.completed_at, "Failed ");
      handler = () => onSelectResult(item.job_id); // Allow viewing failed job details
      classes = "opacity-60 cursor-not-allowed";
      break;
    default:
      icon = <AlertTriangle className="h-5 w-5 text-yellow-500" />;
      text = item.job_id;
      date = "Unknown status";
      handler = null;
      classes = "opacity-50 cursor-not-allowed";
  }

  return (
    <li
      className={`flex items-center justify-between p-4 ${classes} transition-colors`}
    >
      {/* Main clickable area for navigation */}
      <div onClick={handler} className="flex-grow flex items-center space-x-3 cursor-pointer">
        {icon}
        <div className="flex-grow">
          <p className="text-sm font-medium text-white">{text}</p>
          <p className="text-sm text-gray-400">{date}</p>
        </div>
      </div>
      <div className="flex items-center space-x-4">
        <a
          href={`/api/v1/results/${item.job_id}`}
          download={`${item.job_id}.json`}
          onClick={(e) => e.stopPropagation()}
          title="Download Result JSON"
          className="text-gray-500 hover:text-cyan-500 p-1 rounded-full transition-colors"
        >
          <Download className="h-5 w-5" />
        </a>
        <button
          onClick={handleDeleteClick}
          title="Delete Job"
          className="text-gray-500 hover:text-red-500 p-1 rounded-full transition-colors"
        >
          <Trash2 className="h-5 w-5" />
        </button>
      </div>
    </li>
  );
};


/*
================================================================================
Page: Result Detail (ResultDetailPage)
================================================================================
*/
// ... (ResultDetailPage remains unchanged) ...
const ResultDetailPage = ({ resultId, onBack, onVisualize }) => {
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!resultId) {
      setError("No result ID specified.");
      setIsLoading(false);
      return;
    }
    const fetchResult = async () => {
      setIsLoading(true); setError(null);
      try {
        const response = await fetch(`/api/v1/results/${resultId}`);
        if (!response.ok) {
          const data = await response.json();
          throw new Error(data.detail || "Failed to fetch result.");
        }
        const data = await response.json();
        setResult(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setIsLoading(false);
      }
    };
    fetchResult();
  }, [resultId]);

  if (isLoading) return <FullPageLoader />;
  if (error) return <ErrorDisplay error={error} />;
  if (!result) return <ErrorDisplay error="Result data could not be loaded." />;

  // For QUBO, we now also check that the solutions array is not empty.
  const canVisualize = result.residue_selections_mapping && 
                       (result.analysis_type === 'static' && typeof result.results === 'object') ||
                       (result.analysis_type === 'qubo' && result.params?.target_selection_string && result.results?.solutions?.length > 0);

  return (
    <div className="max-w-4xl mx-auto">
      <button
        onClick={onBack}
        className="flex items-center space-x-2 text-cyan-400 hover:text-cyan-300 mb-4"
      >
        <ArrowLeft className="h-5 w-5" />
        <span>Back to Results List</span>
      </button>

      <div className="bg-gray-800 rounded-lg border border-gray-700 shadow-lg p-6">
        <div className="flex justify-between items-start mb-4">
          <div>
            <h2 className="text-2xl font-bold text-white mb-2">Analysis Result</h2>
            <p className="text-sm text-gray-400 mb-1">
              <span className="font-semibold">Job ID:</span> {result.job_id}
            </p>
            <p className="text-sm text-gray-400">
              <span className="font-semibold">Type:</span> <span className="capitalize">{result.analysis_type}</span>
            </p>
          </div>
          {canVisualize && (
            <button
              onClick={() => onVisualize(result.job_id)}
              className="flex items-center space-x-2 bg-cyan-600 hover:bg-cyan-700 text-white font-bold py-2 px-4 rounded-lg transition-colors"
            >
              <Eye className="h-5 w-5" />
              <span>Visualize</span>
            </button>
          )}
        </div>
        
        {result.analysis_type === 'qubo' && (!result.results?.solutions || result.results.solutions.length === 0) && (
          <div className="bg-yellow-900 border border-yellow-700 text-yellow-100 p-3 rounded-md flex items-center space-x-2 my-4">
            <AlertTriangle className="h-5 w-5" />
            <span>No valid solutions with negative energy were found. Visualization is disabled.</span>
          </div>
        )}

        <h3 className="text-lg font-semibold text-cyan-400 mb-2">Raw Result Data</h3>
        <pre className="bg-gray-900 p-4 rounded-md text-gray-200 text-xs overflow-auto">
          {JSON.stringify(result, null, 2)}
        </pre>
      </div>
    </div>
  );
};


/*
================================================================================
Page: Visualize Result (VisualizeResultPage) - (HEAVILY MODIFIED)
================================================================================
*/

/**
 * Helper to build an NGL-compatible selection string from residue keys
 * and the selection mapping.
 */
const getNglSelection = (keys, mapping) => {
  if (!keys || !mapping) return 'none';
  
  const selectionStrings = keys
    .map(key => mapping[key]) // 'res_50' -> 'resid 50'
    .filter(Boolean);         // Filter out any undefined mappings
    
  // Convert 'resid 50' to '50'
  // Convert 'resid 131 22' to '131 or 22'
  const nglResidueNumbers = selectionStrings
    .map(sel => sel.replace(/resid /gi, '')) // '50', '131 22'
    .join(' or '); // '50 or 131 22'
    
  // Final cleanup for NGL: '131 22' -> '131 or 22'
  const finalSelection = nglResidueNumbers.replace(/\s+/g, ' or ') || 'none';
  // console.log(`Keys: ${keys}, NGL Selection: ${finalSelection}`);
  return finalSelection;
};

/**
 * Helper to parse the simple target string (e.g., "resid 131 140")
 * directly into an NGL selection.
 */
const getNglSelectionFromTargetString = (targetString) => {
  if (!targetString) return 'none';
  // "resid 131 140" -> "131 140" -> "131 or 140"
  const selection = targetString.replace(/resid /gi, '');
  return selection.replace(/\s+/g, ' or ') || 'none';
};


const VisualizeResultPage = ({ resultId, onBack }) => {
  const [resultData, setResultData] = useState(null);
  const [structureFile, setStructureFile] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // --- FIX 2: Use useState for NGL stage, not useRef ---
  // This ensures React's effects run when the stage is ready.
  const [nglStage, setNglStage] = useState(null);
  const nglViewportRef = useRef(null);
  // --- END FIX 2 ---
  
  const [staticThreshold, setStaticThreshold] = useState(0.8);
  const [quboSolutionIndex, setQuboSolutionIndex] = useState(0);
  
  // NGL script loader (unchanged)
  useEffect(() => {
    const nglScriptId = 'ngl-script';
    if (document.getElementById(nglScriptId)) {
      return;
    }
    const script = document.createElement('script');
    script.id = nglScriptId;
    script.src = 'https://cdn.jsdelivr.net/npm/ngl/dist/ngl.js';
    script.async = true;
    document.body.appendChild(script);
  }, []);

  // Result data fetcher (unchanged)
  useEffect(() => {
    if (!resultId) {
      setError("No result ID specified.");
      setIsLoading(false);
      return;
    }
    const fetchResult = async () => {
      setIsLoading(true); setError(null);
      try {
        const response = await fetch(`/api/v1/results/${resultId}`);
        if (!response.ok) {
          const data = await response.json();
          throw new Error(data.detail || "Failed to fetch result.");
        }
        const data = await response.json();
        if (!data.residue_selections_mapping || !data.results) {
          throw new Error("Result data is missing the required 'residue_selections_mapping' or 'results' fields for visualization.");
        }
        setResultData(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setIsLoading(false);
      }
    };
    fetchResult();
  }, [resultId]);
  
  // NGL Initialization Effect
  useEffect(() => {
    // Wait for NGL, the structure file, the DOM element, AND for the stage to NOT be set
    if (window.NGL && structureFile && nglViewportRef.current && !nglStage) {
      console.log("Initializing NGL stage...");
      const stage = new window.NGL.Stage(nglViewportRef.current);
      // --- FIX 2: Store stage in state ---
      setNglStage(stage);
      // --- END FIX 2 ---
      
      const ext = structureFile.name.split('.').pop();
      stage.loadFile(structureFile, { ext: ext }).then((component) => {
        // Base protein (greyed out)
        component.addRepresentation("cartoon", { 
          color: '#555555', // Dark grey
          opacity: 0.3
        });
        
        // Representation for TARGET residues (e.g., QUBO target)
        component.addRepresentation("ball+stick", {
          name: "target_highlight",
          sele: "none",
          color: "#3b82f6", // Blue-500
        });
        
        // Representation for SELECTED residues (e.g., Static reporters or QUBO solution)
        component.addRepresentation("ball+stick", {
          name: "selected_highlight",
          sele: "none",
          color: "#ef4444", // Red-500
        });
        
        component.autoView();
      });
    }
    
    // Cleanup
    return () => {
      // --- FIX 2: Use state variable for cleanup ---
      if (nglStage) {
        console.log("Disposing NGL stage");
        nglStage.dispose();
        setNglStage(null);
      }
      // --- END FIX 2 ---
    };
  }, [structureFile, nglStage]); // Re-run if the structure file changes or stage is manually cleared
  
  // NGL Update Effect
  useEffect(() => {
    // --- FIX 2: Wait for state variable, not ref ---
    if (!nglStage || !resultData) return;
    // --- END FIX 2 ---
    
    // --- FIX 1: Correctly destructure 'params' not 'parameters' ---
    const { analysis_type, results, residue_selections_mapping, params } = resultData;
    
    const targetRep = nglStage.getRepresentationsByName('target_highlight');
    const selectedRep = nglStage.getRepresentationsByName('selected_highlight');
    
    if (!targetRep || !selectedRep) return;

    let targetSele = "none";
    let selectedSele = "none";

    if (analysis_type === 'static') {
      // Static Analysis Logic (unchanged)
      const highScoringKeys = Object.keys(results).filter(
        key => results[key] >= staticThreshold
      );
      selectedSele = getNglSelection(highScoringKeys, residue_selections_mapping);
      
    } else if (analysis_type === 'qubo') {
      // QUBO Analysis Logic
      if (!results.solutions || !params || !params.target_selection_string) {
        console.error("QUBO result is missing 'solutions' or 'params.target_selection_string'");
        return;
      }
      
      // --- FIX 3: Use 'params.target_selection_string' directly, as you suggested ---
      targetSele = getNglSelectionFromTargetString(params.target_selection_string);
      // --- END FIX 3 ---

      // 2. Get Selected Residues R_i from the chosen solution
      const solution = results.solutions[quboSolutionIndex];
      if (solution) {
        const selectedKeys = solution.selected_residues;
        selectedSele = getNglSelection(selectedKeys, residue_selections_mapping);
      }
    }
    
    // Apply the selections
    targetRep.setSelection(targetSele);
    selectedRep.setSelection(selectedSele);
    
  // --- FIX 2: Use state variable in dependency array ---
  }, [resultData, nglStage, staticThreshold, quboSolutionIndex]);
  // --- END FIX 2 ---

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      // If we are changing the file, dispose the old stage first
      // --- FIX 2: Use state variable ---
      if (nglStage) {
        nglStage.dispose();
        setNglStage(null);
      }
      // --- END FIX 2 ---
      setStructureFile(file);
      setError(null);
    }
  };

  if (isLoading) return <FullPageLoader />;
  
  return (
    <div className="max-w-7xl mx-auto">
      <button
        onClick={onBack}
        className="flex items-center space-x-2 text-cyan-400 hover:text-cyan-300 mb-4"
      >
        <ArrowLeft className="h-5 w-5" />
        <span>Back to Result Detail</span>
      </button>

      <div className="bg-gray-800 rounded-lg border border-gray-700 shadow-lg p-6">
        <h2 className="text-2xl font-bold text-white mb-4">Visualize Result: {resultId}</h2>
        
        {error && <ErrorDisplay error={error} />}

        {!structureFile ? (
          <div className="text-center bg-gray-900 p-8 rounded-lg border-2 border-dashed border-gray-600">
            <h3 className="text-xl font-semibold text-white mb-4">Upload Structure File</h3>
            <p className="text-gray-400 mb-6">Please upload the PDB or GRO file you used for this analysis.</p>
            <FileDropzone 
              name="structure_file"
              label="Structure (PDB, GRO, ...)"
              file={structureFile}
              onChange={handleFileChange}
            />
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* NGL Viewport */}
            <div className="lg:col-span-2 bg-black rounded-lg h-96 w-full relative">
              <div ref={nglViewportRef} style={{ width: '100%', height: '100%' }} />
              {/* Legend */}
              <div className="absolute top-2 left-2 bg-gray-900 bg-opacity-70 p-2 rounded-md text-xs text-white">
                <h4 className="font-bold mb-1">Legend</h4>
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                  <span>Target (S)</span>
                </div>
                <div className="flex items-center space-x-2 mt-1">
                  <div className="w-3 h-3 rounded-full bg-red-500"></div>
                  <span>Selected (R)</span>
                </div>
              </div>
            </div>
            
            {/* Controls */}
            <div className="bg-gray-900 p-4 rounded-lg">
              <VisualizationControls
                resultData={resultData}
                staticThreshold={staticThreshold}
                setStaticThreshold={setStaticThreshold}
                quboSolutionIndex={quboSolutionIndex}
                setQuboSolutionIndex={setQuboSolutionIndex}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// --- VisualizationControls Sub-component ---
// ... (This component remains unchanged, but is now correct
//      due to the fixes in its parent) ...
const VisualizationControls = ({ 
  resultData, 
  staticThreshold, 
  setStaticThreshold, 
  quboSolutionIndex, 
  setQuboSolutionIndex 
}) => {
  if (!resultData) return null;

  const { analysis_type, results } = resultData;

  if (analysis_type === 'static') {
    return (
      <>
        <h3 className="text-lg font-semibold text-cyan-400 mb-4 flex items-center space-x-2">
          <Palette className="h-5 w-5" />
          <span>Static Reporter Controls</span>
        </h3>
        <div className="space-y-4">
          <div>
            <label htmlFor="threshold" className="block text-sm font-medium text-gray-300 mb-1">
              Highlight Threshold (Accuracy)
            </label>
            <input
              type="range"
              id="threshold"
              min="0"
              max="1"
              step="0.05"
              value={staticThreshold}
              onChange={(e) => setStaticThreshold(parseFloat(e.target.value))}
              className="w-full"
            />
            <div className="text-center text-cyan-400 font-mono text-lg">{staticThreshold.toFixed(2)}</div>
          </div>
          <div className="text-xs text-gray-400">
            Showing residues with an score $\ge$ {staticThreshold.toFixed(2)}.
          </div>
        </div>
      </>
    );
  }

  if (analysis_type === 'qubo') {
    if (!results.solutions || results.solutions.length === 0) {
      return <ErrorDisplay error="QUBO result has no solutions to display." />;
    }
    return (
      <>
        <h3 className="text-lg font-semibold text-cyan-400 mb-4 flex items-center space-x-2">
          <Target className="h-5 w-5" />
          <span>QUBO Solution Controls</span>
        </h3>
        <div className="space-y-4">
          <div>
            <label htmlFor="qubo_solution" className="block text-sm font-medium text-gray-300 mb-1">
              Select Solution
            </label>
            <select
              id="qubo_solution"
              value={quboSolutionIndex}
              onChange={(e) => setQuboSolutionIndex(parseInt(e.target.value))}
              className="w-full bg-gray-800 border border-gray-600 rounded-md p-2 text-white focus:ring-cyan-500 focus:border-cyan-500"
            >
              {results.solutions.map((solution, index) => (
                <option key={index} value={index}>
                  Solution {index + 1} (Energy: {solution.energy.toFixed(4)})
                </option>
              ))}
            </select>
          </div>
          <div className="text-xs text-gray-400">
            <p><span className="text-blue-400 font-bold">Blue:</span> Target residues (S)</p>
            <p><span className="text-red-400 font-bold">Red:</span> Selected residues (R) for this solution</p>
          </div>
          <div className="bg-gray-800 p-2 rounded-md text-xs">
            <p className="font-bold text-gray-300">Selected Residues:</p>
            <p className="text-red-400 break-all">
              {results.solutions[quboSolutionIndex]?.selected_residues.join(', ') || 'None'}
            </p>
          </div>
        </div>
      </>
    );
  }

  // Fallback for other types
  return <p className="text-gray-400">No visualization controls available for this analysis type.</p>;
};



/*
================================================================================
Page: System Health (HealthCheckPage)
================================================================================
*/
// ... (HealthCheckPage and its sub-components remain unchanged) ...
const HealthCheckPage = () => {
  const [healthReport, setHealthReport] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const runHealthCheck = async () => {
    setIsLoading(true); setError(null); setHealthReport(null);
    try {
      const response = await fetch('/api/v1/health/check');
      const data = await response.json();
      if (!response.ok) {
        const detail = data.detail || data;
        setError("System status is not fully OK. See details below.");
        setHealthReport(detail);
      } else {
        setHealthReport(data);
      }
    } catch (err) {
      setError("An unknown error occurred while contacting the server.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto">
      <h2 className="text-3xl font-bold mb-6 text-white">System Health Check</h2>
      <p className="mb-6 text-gray-300">
        Run an end-to-end test of the system, from the API to Redis
        to the background Worker.
      </p>
      <button
        onClick={runHealthCheck} disabled={isLoading}
        className="flex items-center justify-center space-x-2 bg-cyan-600 hover:bg-cyan-700 text-white font-bold py-3 px-6 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed mb-8"
      >
        {isLoading ? <Loader2 className="h-5 w-5 animate-spin" /> : <Activity className="h-5 w-5" />}
        <span>{isLoading ? 'Running Check...' : 'Run E2E Health Check'}</span>
      </button>
      {error && !healthReport && <ErrorDisplay error={error} />}
      {healthReport && (
        <div className="space-y-4">
          <HealthStatusCard title="API Status" status={healthReport.api_status} />
          <HealthStatusCard title="Redis Status" status={healthReport.redis_status?.status} details={healthReport.redis_status} />
          <HealthStatusCard title="Worker Status" status={healthReport.worker_status?.status} details={healthReport.worker_status} />
        </div>
      )}
    </div>
  );
};

const HealthStatusCard = ({ title, status, details }) => {
  const isOk = status === 'ok';
  const displayStatus = status || 'unknown';
  const getStatusIcon = () => isOk ? <CheckCircle className="h-8 w-8 text-green-500" /> : <AlertTriangle className="h-8 w-8 text-yellow-500" />;

  return (
    <div className={`bg-gray-800 rounded-lg border ${isOk ? 'border-green-700' : 'border-yellow-700'} shadow-lg p-6`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          {getStatusIcon()}
          <h3 className="text-xl font-semibold text-white">{title}</h3>
        </div>
        <span className={`px-3 py-1 rounded-full text-sm font-bold ${isOk ? 'bg-green-800 text-green-100' : 'bg-yellow-800 text-yellow-100'}`}>
          {displayStatus}
        </span>
      </div>
      {details && details.status !== 'ok' && (
        <pre className="mt-4 bg-gray-900 p-4 rounded-md text-red-300 text-xs overflow-auto">
          {JSON.stringify(details, null, 2)}
        </pre>
      )}
    </div>
  );
};