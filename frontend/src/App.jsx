import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import AppLayout from './components/layout/AppLayout';
import ProjectsPage from './pages/ProjectsPage';
import SystemDetailPage from './pages/SystemDetailPage';
import ResultsPage from './pages/ResultsPage';
import ResultDetailPage from './pages/ResultDetailPage';
import VisualizePage from './pages/VisualizePage';
import SimulationResultPage from './pages/SimulationResultPage';
import SimulationComparePage from './pages/SimulationComparePage';
import DescriptorVizPage from './pages/DescriptorVizPage';
import MetastableVizPage from './pages/MetastableVizPage';
import SamplingVizPage from './pages/SamplingVizPage';
import DeltaEvalPage from './pages/DeltaEvalPage';
import MolstarDebugPage from './pages/MolstarDebugPage';
import HealthPage from './pages/HealthPage';
import JobStatusPage from './pages/JobStatusPage';

export default function App() {
  return (
    <BrowserRouter>
      <AppLayout>
        <Routes>
          <Route path="/" element={<Navigate to="/projects" replace />} />
          <Route path="/projects" element={<ProjectsPage />} />
          <Route path="/projects/:projectId/systems/:systemId" element={<SystemDetailPage />} />
          <Route
            path="/projects/:projectId/systems/:systemId/descriptors/visualize"
            element={<DescriptorVizPage />}
          />
          <Route
            path="/projects/:projectId/systems/:systemId/metastable/visualize"
            element={<MetastableVizPage />}
          />
          <Route
            path="/projects/:projectId/systems/:systemId/sampling/visualize"
            element={<SamplingVizPage />}
          />
          <Route
            path="/projects/:projectId/systems/:systemId/sampling/delta_eval"
            element={<DeltaEvalPage />}
          />
          <Route path="/results" element={<ResultsPage />} />
          <Route path="/results/:jobId" element={<ResultDetailPage />} />
          <Route path="/visualize/:jobId" element={<VisualizePage />} />
          <Route path="/simulation/compare" element={<SimulationComparePage />} />
          <Route path="/simulation/:jobId" element={<SimulationResultPage />} />
          <Route path="/health" element={<HealthPage />} />
          <Route path="/jobs/:jobId" element={<JobStatusPage />} />
          <Route path="/debug/molstar" element={<MolstarDebugPage />} />
          <Route path="*" element={<Navigate to="/projects" replace />} />
        </Routes>
      </AppLayout>
    </BrowserRouter>
  );
}
