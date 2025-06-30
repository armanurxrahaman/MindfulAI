import React, { createContext, useContext, useState } from 'react';
import ReactDOM from 'react-dom/client';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import App from './App';

// Create theme context
const ThemeContext = createContext();

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};

// Light theme
const lightTheme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#667eea', // Spiritual indigo
      light: '#a3bffa',
      dark: '#4c51bf',
    },
    secondary: {
      main: '#f093fb', // Soft lavender
      light: '#fbb6ce',
      dark: '#d53f8c',
    },
    success: {
      main: '#68d391', // Healing green
      light: '#9ae6b4',
      dark: '#38a169',
    },
    warning: {
      main: '#f6ad55', // Warm orange
      light: '#fbd38d',
      dark: '#dd6b20',
    },
    error: {
      main: '#fc8181', // Soft red
      light: '#fed7d7',
      dark: '#e53e3e',
    },
    background: {
      default: '#fafbfc',
      paper: '#ffffff',
    },
    text: {
      primary: '#2d3748',
      secondary: '#718096',
    },
    grey: {
      50: '#f7fafc',
      100: '#edf2f7',
      200: '#e2e8f0',
      300: '#cbd5e0',
      400: '#a0aec0',
      500: '#718096',
      600: '#4a5568',
      700: '#2d3748',
      800: '#1a202c',
      900: '#171923',
    },
  },
  typography: {
    fontFamily: [
      'Inter',
      'Poppins',
      'Quicksand',
      '-apple-system',
      'BlinkMacSystemFont',
      '"Segoe UI"',
      'Roboto',
      '"Helvetica Neue"',
      'Arial',
      'sans-serif',
    ].join(','),
    h1: {
      fontFamily: 'Quicksand, sans-serif',
      fontWeight: 700,
      fontSize: '3.5rem',
      lineHeight: 1.2,
      letterSpacing: '-0.02em',
      background: 'linear-gradient(135deg, #667eea 0%, #f093fb 100%)',
      backgroundClip: 'text',
      WebkitBackgroundClip: 'text',
      WebkitTextFillColor: 'transparent',
    },
    h2: {
      fontFamily: 'Quicksand, sans-serif',
      fontWeight: 600,
      fontSize: '2.5rem',
      lineHeight: 1.3,
      letterSpacing: '-0.01em',
    },
    h3: {
      fontFamily: 'Quicksand, sans-serif',
      fontWeight: 600,
      fontSize: '2rem',
      lineHeight: 1.4,
    },
    h4: {
      fontFamily: 'Inter, sans-serif',
      fontWeight: 600,
      fontSize: '1.5rem',
      lineHeight: 1.4,
    },
    h5: {
      fontFamily: 'Inter, sans-serif',
      fontWeight: 600,
      fontSize: '1.25rem',
      lineHeight: 1.4,
    },
    h6: {
      fontFamily: 'Inter, sans-serif',
      fontWeight: 600,
      fontSize: '1rem',
      lineHeight: 1.4,
    },
    body1: {
      fontFamily: 'Inter, sans-serif',
      fontSize: '1rem',
      lineHeight: 1.6,
      fontWeight: 400,
    },
    body2: {
      fontFamily: 'Inter, sans-serif',
      fontSize: '0.875rem',
      lineHeight: 1.6,
      fontWeight: 400,
    },
    button: {
      fontFamily: 'Inter, sans-serif',
      fontWeight: 600,
      textTransform: 'none',
      letterSpacing: '0.025em',
    },
    subtitle1: {
      fontFamily: 'Inter, sans-serif',
      fontWeight: 500,
      fontSize: '1rem',
      lineHeight: 1.6,
    },
    subtitle2: {
      fontFamily: 'Inter, sans-serif',
      fontWeight: 500,
      fontSize: '0.875rem',
      lineHeight: 1.6,
    },
  },
  shape: {
    borderRadius: 16,
  },
  shadows: [
    'none',
    '0px 2px 4px rgba(0, 0, 0, 0.05)',
    '0px 4px 8px rgba(0, 0, 0, 0.05)',
    '0px 8px 16px rgba(0, 0, 0, 0.05)',
    '0px 16px 32px rgba(0, 0, 0, 0.05)',
    '0px 32px 64px rgba(0, 0, 0, 0.05)',
    '0px 2px 4px rgba(0, 0, 0, 0.05)',
    '0px 4px 8px rgba(0, 0, 0, 0.05)',
    '0px 8px 16px rgba(0, 0, 0, 0.05)',
    '0px 16px 32px rgba(0, 0, 0, 0.05)',
    '0px 32px 64px rgba(0, 0, 0, 0.05)',
    '0px 2px 4px rgba(0, 0, 0, 0.05)',
    '0px 4px 8px rgba(0, 0, 0, 0.05)',
    '0px 8px 16px rgba(0, 0, 0, 0.05)',
    '0px 16px 32px rgba(0, 0, 0, 0.05)',
    '0px 32px 64px rgba(0, 0, 0, 0.05)',
    '0px 2px 4px rgba(0, 0, 0, 0.05)',
    '0px 4px 8px rgba(0, 0, 0, 0.05)',
    '0px 8px 16px rgba(0, 0, 0, 0.05)',
    '0px 16px 32px rgba(0, 0, 0, 0.05)',
    '0px 32px 64px rgba(0, 0, 0, 0.05)',
    '0px 2px 4px rgba(0, 0, 0, 0.05)',
    '0px 4px 8px rgba(0, 0, 0, 0.05)',
    '0px 8px 16px rgba(0, 0, 0, 0.05)',
    '0px 16px 32px rgba(0, 0, 0, 0.05)',
    '0px 32px 64px rgba(0, 0, 0, 0.05)',
  ],
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 12,
          padding: '12px 32px',
          fontWeight: 600,
          fontSize: '0.95rem',
          letterSpacing: '0.025em',
          transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
          '&:hover': {
            transform: 'translateY(-2px)',
            boxShadow: '0 8px 25px rgba(0, 0, 0, 0.15)',
          },
        },
        contained: {
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          '&:hover': {
            background: 'linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%)',
          },
        },
        outlined: {
          borderWidth: '2px',
          '&:hover': {
            borderWidth: '2px',
            transform: 'translateY(-2px)',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.08)',
          border: '1px solid rgba(255, 255, 255, 0.2)',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 20,
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.08)',
          transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
          '&:hover': {
            transform: 'translateY(-8px)',
            boxShadow: '0 20px 40px rgba(0, 0, 0, 0.12)',
          },
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 12,
            transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
            '&:hover': {
              transform: 'translateY(-1px)',
            },
            '&.Mui-focused': {
              transform: 'translateY(-1px)',
            },
          },
        },
      },
    },
    MuiTabs: {
      styleOverrides: {
        root: {
          '& .MuiTab-root': {
            borderRadius: 12,
            margin: '0 8px',
            transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
            '&:hover': {
              backgroundColor: 'rgba(102, 126, 234, 0.08)',
            },
          },
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          background: 'rgba(255, 255, 255, 0.95)',
          backdropFilter: 'blur(20px)',
          borderBottom: '1px solid rgba(255, 255, 255, 0.2)',
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.08)',
        },
      },
    },
  },
});

// Dark theme
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#a3bffa', // Lighter spiritual indigo for dark mode
      light: '#c6f6d5',
      dark: '#667eea',
    },
    secondary: {
      main: '#fbb6ce', // Lighter lavender for dark mode
      light: '#fed7e2',
      dark: '#f093fb',
    },
    success: {
      main: '#9ae6b4', // Lighter healing green
      light: '#c6f6d5',
      dark: '#68d391',
    },
    warning: {
      main: '#fbd38d', // Lighter warm orange
      light: '#fef5e7',
      dark: '#f6ad55',
    },
    error: {
      main: '#fed7d7', // Lighter soft red
      light: '#fff5f5',
      dark: '#fc8181',
    },
    background: {
      default: '#0f1419', // Deep spiritual dark
      paper: '#1a202c',
    },
    text: {
      primary: '#f7fafc',
      secondary: '#a0aec0',
    },
    grey: {
      50: '#171923',
      100: '#1a202c',
      200: '#2d3748',
      300: '#4a5568',
      400: '#718096',
      500: '#a0aec0',
      600: '#cbd5e0',
      700: '#e2e8f0',
      800: '#edf2f7',
      900: '#f7fafc',
    },
  },
  typography: {
    fontFamily: [
      'Inter',
      'Poppins',
      'Quicksand',
      '-apple-system',
      'BlinkMacSystemFont',
      '"Segoe UI"',
      'Roboto',
      '"Helvetica Neue"',
      'Arial',
      'sans-serif',
    ].join(','),
    h1: {
      fontFamily: 'Quicksand, sans-serif',
      fontWeight: 700,
      fontSize: '3.5rem',
      lineHeight: 1.2,
      letterSpacing: '-0.02em',
      background: 'linear-gradient(135deg, #a3bffa 0%, #fbb6ce 100%)',
      backgroundClip: 'text',
      WebkitBackgroundClip: 'text',
      WebkitTextFillColor: 'transparent',
    },
    h2: {
      fontFamily: 'Quicksand, sans-serif',
      fontWeight: 600,
      fontSize: '2.5rem',
      lineHeight: 1.3,
      letterSpacing: '-0.01em',
    },
    h3: {
      fontFamily: 'Quicksand, sans-serif',
      fontWeight: 600,
      fontSize: '2rem',
      lineHeight: 1.4,
    },
    h4: {
      fontFamily: 'Inter, sans-serif',
      fontWeight: 600,
      fontSize: '1.5rem',
      lineHeight: 1.4,
    },
    h5: {
      fontFamily: 'Inter, sans-serif',
      fontWeight: 600,
      fontSize: '1.25rem',
      lineHeight: 1.4,
    },
    h6: {
      fontFamily: 'Inter, sans-serif',
      fontWeight: 600,
      fontSize: '1rem',
      lineHeight: 1.4,
    },
    body1: {
      fontFamily: 'Inter, sans-serif',
      fontSize: '1rem',
      lineHeight: 1.6,
      fontWeight: 400,
    },
    body2: {
      fontFamily: 'Inter, sans-serif',
      fontSize: '0.875rem',
      lineHeight: 1.6,
      fontWeight: 400,
    },
    button: {
      fontFamily: 'Inter, sans-serif',
      fontWeight: 600,
      textTransform: 'none',
      letterSpacing: '0.025em',
    },
    subtitle1: {
      fontFamily: 'Inter, sans-serif',
      fontWeight: 500,
      fontSize: '1rem',
      lineHeight: 1.6,
    },
    subtitle2: {
      fontFamily: 'Inter, sans-serif',
      fontWeight: 500,
      fontSize: '0.875rem',
      lineHeight: 1.6,
    },
  },
  shape: {
    borderRadius: 16,
  },
  shadows: [
    'none',
    '0px 2px 4px rgba(0, 0, 0, 0.3)',
    '0px 4px 8px rgba(0, 0, 0, 0.3)',
    '0px 8px 16px rgba(0, 0, 0, 0.3)',
    '0px 16px 32px rgba(0, 0, 0, 0.3)',
    '0px 32px 64px rgba(0, 0, 0, 0.3)',
    '0px 2px 4px rgba(0, 0, 0, 0.3)',
    '0px 4px 8px rgba(0, 0, 0, 0.3)',
    '0px 8px 16px rgba(0, 0, 0, 0.3)',
    '0px 16px 32px rgba(0, 0, 0, 0.3)',
    '0px 32px 64px rgba(0, 0, 0, 0.3)',
    '0px 2px 4px rgba(0, 0, 0, 0.3)',
    '0px 4px 8px rgba(0, 0, 0, 0.3)',
    '0px 8px 16px rgba(0, 0, 0, 0.3)',
    '0px 16px 32px rgba(0, 0, 0, 0.3)',
    '0px 32px 64px rgba(0, 0, 0, 0.3)',
    '0px 2px 4px rgba(0, 0, 0, 0.3)',
    '0px 4px 8px rgba(0, 0, 0, 0.3)',
    '0px 8px 16px rgba(0, 0, 0, 0.3)',
    '0px 16px 32px rgba(0, 0, 0, 0.3)',
    '0px 32px 64px rgba(0, 0, 0, 0.3)',
    '0px 2px 4px rgba(0, 0, 0, 0.3)',
    '0px 4px 8px rgba(0, 0, 0, 0.3)',
    '0px 8px 16px rgba(0, 0, 0, 0.3)',
    '0px 16px 32px rgba(0, 0, 0, 0.3)',
    '0px 32px 64px rgba(0, 0, 0, 0.3)',
  ],
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 12,
          padding: '12px 32px',
          fontWeight: 600,
          fontSize: '0.95rem',
          letterSpacing: '0.025em',
          transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
          '&:hover': {
            transform: 'translateY(-2px)',
            boxShadow: '0 8px 25px rgba(0, 0, 0, 0.4)',
          },
        },
        contained: {
          background: 'linear-gradient(135deg, #a3bffa 0%, #fbb6ce 100%)',
          color: '#1a202c',
          '&:hover': {
            background: 'linear-gradient(135deg, #c6f6d5 0%, #fed7e2 100%)',
          },
        },
        outlined: {
          borderWidth: '2px',
          '&:hover': {
            borderWidth: '2px',
            transform: 'translateY(-2px)',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.3)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 20,
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.3)',
          transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
          '&:hover': {
            transform: 'translateY(-8px)',
            boxShadow: '0 20px 40px rgba(0, 0, 0, 0.4)',
          },
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 12,
            transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
            '&:hover': {
              transform: 'translateY(-1px)',
            },
            '&.Mui-focused': {
              transform: 'translateY(-1px)',
            },
          },
        },
      },
    },
    MuiTabs: {
      styleOverrides: {
        root: {
          '& .MuiTab-root': {
            borderRadius: 12,
            margin: '0 8px',
            transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
            '&:hover': {
              backgroundColor: 'rgba(163, 191, 250, 0.1)',
            },
          },
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          background: 'rgba(26, 32, 44, 0.95)',
          backdropFilter: 'blur(20px)',
          borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.3)',
        },
      },
    },
  },
});

// Theme provider component
const ThemeProviderWrapper = ({ children }) => {
  const [isDarkMode, setIsDarkMode] = useState(false);
  
  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode);
  };
  
  const theme = isDarkMode ? darkTheme : lightTheme;
  
  return (
    <ThemeContext.Provider value={{ isDarkMode, toggleTheme }}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        {children}
      </ThemeProvider>
    </ThemeContext.Provider>
  );
};

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <ThemeProviderWrapper>
      <App />
    </ThemeProviderWrapper>
  </React.StrictMode>
); 