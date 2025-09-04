import React from 'react';
import styled from 'styled-components';
import { Link, useLocation, useNavigate } from 'react-router-dom';

const NavContainer = styled.nav`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 2rem;
  background: rgba(24, 26, 32, 0.95);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  z-index: 1000;
`;

const Logo = styled.div`
  font-size: 1.5rem;
  font-weight: 800;
  color: #fff;
  font-family: 'Inter', sans-serif;
  letter-spacing: 0.1em;
  cursor: pointer;
  transition: all 0.3s ease;
  
  &:hover {
    color: #c084fc;
    transform: scale(1.05);
  }
`;

const NavMenu = styled.ul`
  display: flex;
  list-style: none;
  margin: 0;
  padding: 0;
  gap: 2rem;
`;

const NavItem = styled.li`
  position: relative;
`;

const NavLink = styled(Link)`
  color: ${props => props.$active ? '#c084fc' : '#b0b8c1'};
  text-decoration: none;
  font-size: 0.9rem;
  font-weight: 500;
  font-family: 'Inter', sans-serif;
  padding: 0.5rem 0;
  transition: color 0.3s ease;
  cursor: pointer;
  
  &:hover {
    color: #c084fc;
  }
  
  &::after {
    content: '';
    position: absolute;
    bottom: -2px;
    left: 0;
    width: ${props => props.$active ? '100%' : '0%'};
    height: 2px;
    background: linear-gradient(90deg, #ff6ac1, #38bdf8);
    transition: width 0.3s ease;
  }
  
  &:hover::after {
    width: 100%;
  }
`;

function Navigation() {
  const location = useLocation();
  const navigate = useNavigate();
  const menuMap = {
    '/': 'Home',
    '/raw-data': 'User Manual',
    '/api-generator': 'Content Generator',
  };
  const activeMenu = menuMap[location.pathname] || 'Home';
  
  const handleLogoClick = () => {
    navigate('/');
  };
  
  return (
    <NavContainer>
      <Logo onClick={handleLogoClick}>ASTRA</Logo>
      <NavMenu>
        <NavItem>
          <NavLink to="/" $active={activeMenu === 'Home'}>Home</NavLink>
        </NavItem>
        <NavItem>
          <NavLink to="/raw-data" $active={activeMenu === 'User Manual'}>User Manual</NavLink>
        </NavItem>
        <NavItem>
          <NavLink to="/api-generator" $active={activeMenu === 'Content Generator'}>Content Generator</NavLink>
        </NavItem>
      </NavMenu>
    </NavContainer>
  );
}

export default Navigation; 