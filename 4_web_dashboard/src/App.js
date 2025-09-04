import React, { useEffect, useState } from 'react';
import HeroSection from './components/HeroSection';
import PackedBubbleChart from './components/PackedBubbleChart';
import Navigation from './components/Navigation';
import styled from 'styled-components';
import './App.css';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import RawData from './components/RawData';
import TrendStrategy from './components/TrendStrategy';
import ApiGenerator from './components/ApiGenerator';
import GeneratedResults from './components/GeneratedResults';
import databaseService from './services/databaseService';

const MainContainer = styled.div`
  display: flex;
  flex-direction: row;
  justify-content: center;
  align-items: center;
  width: 100vw;
  min-height: 100vh;
  gap: 2vw;
  padding-top: 60px;
`;

const LeftSection = styled.div`
  min-height: 100vh;
  display: flex;
  align-items: flex-start;
  justify-content: center;
  padding-top: 6vh;
  padding-left: 6vh;
`;

const RightSection = styled.div`
  min-height: 100vh;
  display: flex;
  align-items: flex-start;
  padding-top: 6vh;
  padding-left: 8vw;
`;

const HeroContainer = styled.section`
  margin-top: 4vh;
`;

function App() {
  const [trendData, setTrendData] = useState([]);

              useEffect(() => {
    const loadBubbleChartData = async () => {
      try {
        console.log('Loading bubble chart data...');
        const data = await databaseService.loadBubbleChartData();
        console.log('Bubble chart data loaded:', data);
        setTrendData(data);
      } catch (error) {
        console.error('Error loading bubble chart data:', error);
        setTrendData([]);
      }
    };
    
    loadBubbleChartData();
  }, []);

  return (
    <Router>
      <Navigation />
      <Routes>
        <Route path="/" element={
          <MainContainer>
            <LeftSection>
              <HeroContainer>
                <HeroSection />
              </HeroContainer>
            </LeftSection>
            <RightSection>
              <PackedBubbleChart data={trendData} />
            </RightSection>
          </MainContainer>
        } />
        <Route path="/raw-data" element={<RawData />} />
        <Route path="/trend-strategy" element={<TrendStrategy />} />
        <Route path="/topic/:topicId" element={<TrendStrategy />} />
        <Route path="/api-generator" element={<ApiGenerator />} />
        <Route path="/generated-results" element={<GeneratedResults />} />
        {/* Add more routes as needed */}
      </Routes>
    </Router>
  );
}

export default App;
