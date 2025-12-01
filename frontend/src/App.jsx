import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import AppLayout from './components/layout/AppLayout';
import ProjectsPage from './pages/ProjectsPage';
import SystemDetailPage from './pages/SystemDetailPage';
import ResultsPage from './pages/ResultsPage';
import ResultDetailPage from './pages/ResultDetailPage';
import VisualizePage from './pages/VisualizePage';
import DescriptorVizPage from './pages/DescriptorVizPage';
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
          <Route path="/results" element={<ResultsPage />} />
          <Route path="/results/:jobId" element={<ResultDetailPage />} />
          <Route path="/visualize/:jobId" element={<VisualizePage />} />
          <Route path="/health" element={<HealthPage />} />
          <Route path="/jobs/:jobId" element={<JobStatusPage />} />
          <Route path="/debug/molstar" element={<MolstarDebugPage />} />
          <Route path="*" element={<Navigate to="/projects" replace />} />
        </Routes>
      </AppLayout>
    </BrowserRouter>
  );
}
