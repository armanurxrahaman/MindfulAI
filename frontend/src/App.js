import React, { useState, useEffect } from 'react';
import {
  Container,
  Box,
  Typography,
  Paper,
  Grid,
  TextField,
  Button,
  Card,
  CardContent,
  List,
  ListItem,
  ListItemText,
  CircularProgress,
  Alert,
  useTheme as useMuiTheme,
  alpha,
  AppBar,
  Toolbar,
  IconButton,
  Tabs,
  Tab,
  Chip,
  Avatar,
  Divider,
  Tooltip
} from '@mui/material';
import { 
  CameraAlt, 
  TextFields, 
  Psychology, 
  Favorite, 
  Lightbulb,
  Spa,
  SelfImprovement,
  EmojiEmotions,
  Brightness4,
  Brightness7
} from '@mui/icons-material';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip as ChartTooltip,
  Legend
} from 'chart.js';
import axios from 'axios';
import { useTheme } from './index';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  ChartTooltip,
  Legend
);

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Floating particles component
const FloatingParticles = ({ isDarkMode }) => {
  return (
    <Box
      sx={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        zIndex: 0,
        overflow: 'hidden',
      }}
    >
      {[...Array(20)].map((_, i) => (
        <Box
          key={i}
          sx={{
            position: 'absolute',
            width: Math.random() * 4 + 2,
            height: Math.random() * 4 + 2,
            background: isDarkMode 
              ? `rgba(${Math.random() * 100 + 155}, ${Math.random() * 100 + 180}, ${Math.random() * 100 + 220}, 0.4)`
              : `rgba(${Math.random() * 100 + 155}, ${Math.random() * 100 + 155}, ${Math.random() * 100 + 200}, 0.3)`,
            borderRadius: '50%',
            left: `${Math.random() * 100}%`,
            top: `${Math.random() * 100}%`,
            animation: `float ${Math.random() * 10 + 10}s infinite linear`,
            animationDelay: `${Math.random() * 5}s`,
            '@keyframes float': {
              '0%': {
                transform: 'translateY(0px) rotate(0deg)',
                opacity: 0,
              },
              '10%': {
                opacity: 1,
              },
              '90%': {
                opacity: 1,
              },
              '100%': {
                transform: 'translateY(-100vh) rotate(360deg)',
                opacity: 0,
              },
            },
          }}
        />
      ))}
    </Box>
  );
};

function App() {
  const theme = useMuiTheme();
  const { isDarkMode, toggleTheme } = useTheme();
  const [text, setText] = useState('');
  const [emotionResult, setEmotionResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dailyContent, setDailyContent] = useState([]);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState(0);
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);

  useEffect(() => {
    fetchDailyContent();
  }, []);

  const fetchDailyContent = async () => {
    try {
      const response = await axios.get(`${API_URL}/daily_content`);
      setDailyContent(response.data.content || []);
    } catch (error) {
      console.error('Error fetching daily content:', error);
      setError('Failed to fetch daily content');
    }
  };

  const analyzeText = async () => {
    if (!text.trim()) return;

    setLoading(true);
    setError(null);
    try {
      const response = await axios.post(`${API_URL}/analyze/text`, { text });
      setEmotionResult(response.data);
    } catch (error) {
      console.error('Error analyzing text:', error);
      setError('Failed to analyze text');
    }
    setLoading(false);
  };

  const handleImageSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const analyzeImage = async () => {
    if (!selectedImage) return;

    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('file', selectedImage);

      const response = await axios.post(`${API_URL}/analyze/image`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setEmotionResult(response.data);
    } catch (error) {
      console.error('Error analyzing image:', error);
      if (error.response?.status === 400) {
        setError(error.response.data.detail || 'Failed to analyze image');
      } else {
        setError('Failed to analyze image');
      }
    }
    setLoading(false);
  };

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
    setEmotionResult(null);
    setError(null);
  };

  const getEmotionColor = (emotion) => {
    const colors = {
      happy: '#68d391',
      sad: '#667eea',
      angry: '#fc8181',
      fear: '#f6ad55',
      surprise: '#f093fb',
      disgust: '#9f7aea',
      neutral: '#a0aec0'
    };
    return colors[emotion] || colors.neutral;
  };

  const getEmotionIcon = (emotion) => {
    const icons = {
      happy: <EmojiEmotions sx={{ color: getEmotionColor(emotion) }} />,
      sad: <Psychology sx={{ color: getEmotionColor(emotion) }} />,
      angry: <SelfImprovement sx={{ color: getEmotionColor(emotion) }} />,
      fear: <Spa sx={{ color: getEmotionColor(emotion) }} />,
      surprise: <Lightbulb sx={{ color: getEmotionColor(emotion) }} />,
      disgust: <Psychology sx={{ color: getEmotionColor(emotion) }} />,
      neutral: <EmojiEmotions sx={{ color: getEmotionColor(emotion) }} />
    };
    return icons[emotion] || icons.neutral;
  };

  return (
    <Box sx={{ position: 'relative' }}>
      <FloatingParticles isDarkMode={isDarkMode} />
      
      <AppBar position="static" color="default" elevation={0}>
        <Toolbar>
          <Box sx={{ display: 'flex', alignItems: 'center', flexGrow: 1 }}>
            <Avatar
              sx={{
                background: 'linear-gradient(135deg, #667eea 0%, #f093fb 100%)',
                mr: 2,
                width: 40,
                height: 40
              }}
            >
              <Psychology />
            </Avatar>
            <Typography 
              variant="h6" 
              component="div" 
              sx={{ 
                fontWeight: 700,
                background: 'linear-gradient(135deg, #667eea 0%, #f093fb 100%)',
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent'
              }}
            >
              MindfulAI by Armanur
            </Typography>
          </Box>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Chip 
              label="Your Safe Space" 
              color="primary" 
              variant="outlined"
              sx={{ borderRadius: 20 }}
            />
            
            <Tooltip title={isDarkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}>
              <IconButton
                onClick={toggleTheme}
                sx={{
                  color: theme.palette.text.primary,
                  backgroundColor: alpha(theme.palette.primary.main, 0.1),
                  '&:hover': {
                    backgroundColor: alpha(theme.palette.primary.main, 0.2),
                    transform: 'rotate(180deg)',
                  },
                  transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                }}
              >
                {isDarkMode ? <Brightness7 /> : <Brightness4 />}
              </IconButton>
            </Tooltip>
          </Box>
        </Toolbar>
      </AppBar>

      <Box sx={{ 
        minHeight: '100vh',
        background: isDarkMode 
          ? `linear-gradient(135deg, 
              ${alpha(theme.palette.primary.light, 0.03)} 0%, 
              ${alpha(theme.palette.secondary.light, 0.03)} 50%,
              ${alpha(theme.palette.success.light, 0.02)} 100%)`
          : `linear-gradient(135deg, 
              ${alpha(theme.palette.primary.light, 0.05)} 0%, 
              ${alpha(theme.palette.secondary.light, 0.05)} 50%,
              ${alpha(theme.palette.success.light, 0.03)} 100%)`,
        py: 6,
        position: 'relative',
        zIndex: 1
      }}>
        <Container maxWidth="lg">
          <Box sx={{ mb: 8, textAlign: 'center' }}>
            <Typography 
              variant="h1" 
              component="h1" 
              gutterBottom 
              sx={{ 
                mb: 3,
                textShadow: isDarkMode ? '0 2px 4px rgba(0,0,0,0.3)' : '0 2px 4px rgba(0,0,0,0.1)'
              }}
            >
              Welcome to MindfulAI
            </Typography>
            <Typography 
              variant="h5" 
              color="text.secondary"
              sx={{ 
                maxWidth: '700px', 
                mx: 'auto',
                mb: 4,
                fontWeight: 400,
                lineHeight: 1.6
              }}
            >
              Discover inner peace through mindful emotion analysis and personalized guidance
            </Typography>
            
            <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, flexWrap: 'wrap' }}>
              <Chip 
                icon={<Psychology />} 
                label="Emotion Analysis" 
                color="primary" 
                variant="outlined"
                sx={{ borderRadius: 20 }}
              />
              <Chip 
                icon={<Spa />} 
                label="Mindful Guidance" 
                color="secondary" 
                variant="outlined"
                sx={{ borderRadius: 20 }}
              />
              <Chip 
                icon={<Favorite />} 
                label="Personal Growth" 
                color="success" 
                variant="outlined"
                sx={{ borderRadius: 20 }}
              />
            </Box>
          </Box>

          {error && (
            <Alert 
              severity="error" 
              sx={{ 
                mb: 4,
                borderRadius: 3,
                boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
                backdropFilter: 'blur(10px)'
              }}
            >
              {error}
            </Alert>
          )}

          <Grid container spacing={4}>
            {/* Analysis Section */}
            <Grid item xs={12} lg={6}>
              <Paper 
                sx={{ 
                  p: 5,
                  height: '100%',
                  background: isDarkMode 
                    ? 'rgba(26, 32, 44, 0.95)' 
                    : 'rgba(255, 255, 255, 0.95)',
                  backdropFilter: 'blur(20px)',
                  border: `1px solid ${alpha(theme.palette.primary.main, 0.1)}`,
                  borderRadius: 4,
                  position: 'relative',
                  overflow: 'hidden',
                  '&::before': {
                    content: '""',
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    height: '4px',
                    background: 'linear-gradient(90deg, #667eea 0%, #f093fb 100%)',
                  }
                }}
              >
                <Typography 
                  variant="h4" 
                  gutterBottom
                  sx={{ 
                    color: theme.palette.primary.main,
                    mb: 4,
                    fontWeight: 700,
                    display: 'flex',
                    alignItems: 'center',
                    gap: 2
                  }}
                >
                  <Psychology sx={{ fontSize: 32 }} />
                  How are you feeling today?
                </Typography>

                <Tabs 
                  value={activeTab} 
                  onChange={handleTabChange}
                  sx={{ 
                    mb: 4,
                    '& .MuiTabs-indicator': {
                      background: 'linear-gradient(90deg, #667eea 0%, #f093fb 100%)',
                      height: 3,
                      borderRadius: 2
                    }
                  }}
                >
                  <Tab 
                    icon={<TextFields />} 
                    label="Text Analysis" 
                    iconPosition="start"
                    sx={{ fontWeight: 600 }}
                  />
                  <Tab 
                    icon={<CameraAlt />} 
                    label="Face Analysis" 
                    iconPosition="start"
                    sx={{ fontWeight: 600 }}
                  />
                </Tabs>

                {activeTab === 0 && (
                  <Box>
                    <TextField
                      fullWidth
                      multiline
                      rows={4}
                      value={text}
                      onChange={(e) => setText(e.target.value)}
                      placeholder="Share your thoughts and feelings with us... ✨"
                      sx={{ 
                        mb: 4,
                        '& .MuiOutlinedInput-root': {
                          borderRadius: 3,
                          fontSize: '1.1rem',
                          '&:hover fieldset': {
                            borderColor: theme.palette.primary.main,
                          },
                        },
                      }}
                    />
                    <Button
                      variant="contained"
                      onClick={analyzeText}
                      disabled={loading || !text.trim()}
                      fullWidth
                      size="large"
                      sx={{
                        py: 2,
                        fontSize: '1.1rem',
                        fontWeight: 700,
                        borderRadius: 3,
                        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                        '&:hover': {
                          background: 'linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%)',
                        },
                      }}
                    >
                      {loading ? (
                        <CircularProgress size={28} color="inherit" />
                      ) : (
                        <>
                          <Psychology sx={{ mr: 1 }} />
                          Analyze My Emotions
                        </>
                      )}
                    </Button>
                  </Box>
                )}

                {activeTab === 1 && (
                  <Box>
                    <input
                      accept="image/*"
                      style={{ display: 'none' }}
                      id="image-upload"
                      type="file"
                      onChange={handleImageSelect}
                    />
                    <label htmlFor="image-upload">
                      <Button
                        variant="outlined"
                        component="span"
                        fullWidth
                        size="large"
                        startIcon={<CameraAlt />}
                        sx={{ 
                          mb: 4,
                          py: 3,
                          borderStyle: 'dashed',
                          borderWidth: 2,
                          borderRadius: 3,
                          fontSize: '1.1rem',
                          fontWeight: 600,
                          '&:hover': {
                            borderColor: theme.palette.secondary.main,
                            backgroundColor: alpha(theme.palette.secondary.main, 0.05),
                          },
                        }}
                      >
                        {selectedImage ? 'Change Image' : 'Upload Your Photo'}
                      </Button>
                    </label>

                    {imagePreview && (
                      <Box sx={{ mb: 4, textAlign: 'center' }}>
                        <img 
                          src={imagePreview} 
                          alt="Preview" 
                          style={{ 
                            maxWidth: '100%', 
                            maxHeight: '250px', 
                            borderRadius: '16px',
                            border: `3px solid ${alpha(theme.palette.secondary.main, 0.3)}`,
                            boxShadow: '0 8px 32px rgba(0,0,0,0.1)'
                          }} 
                        />
                      </Box>
                    )}

                    <Button
                      variant="contained"
                      onClick={analyzeImage}
                      disabled={loading || !selectedImage}
                      fullWidth
                      size="large"
                      sx={{
                        py: 2,
                        fontSize: '1.1rem',
                        fontWeight: 700,
                        borderRadius: 3,
                        background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
                        '&:hover': {
                          background: 'linear-gradient(135deg, #d53f8c 0%, #e53e3e 100%)',
                        },
                      }}
                    >
                      {loading ? (
                        <CircularProgress size={28} color="inherit" />
                      ) : (
                        <>
                          <CameraAlt sx={{ mr: 1 }} />
                          Analyze My Expression
                        </>
                      )}
                    </Button>
                  </Box>
                )}
              </Paper>
            </Grid>

            {/* Results Section */}
            <Grid item xs={12} lg={6}>
              <Paper 
                sx={{ 
                  p: 5,
                  height: '100%',
                  background: isDarkMode 
                    ? 'rgba(26, 32, 44, 0.95)' 
                    : 'rgba(255, 255, 255, 0.95)',
                  backdropFilter: 'blur(20px)',
                  border: `1px solid ${alpha(theme.palette.secondary.main, 0.1)}`,
                  borderRadius: 4,
                  position: 'relative',
                  overflow: 'hidden',
                  '&::before': {
                    content: '""',
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    height: '4px',
                    background: 'linear-gradient(90deg, #f093fb 0%, #f5576c 100%)',
                  }
                }}
              >
                <Typography 
                  variant="h4" 
                  gutterBottom
                  sx={{ 
                    color: theme.palette.secondary.main,
                    mb: 4,
                    fontWeight: 700,
                    display: 'flex',
                    alignItems: 'center',
                    gap: 2
                  }}
                >
                  <Lightbulb sx={{ fontSize: 32 }} />
                  Your Emotional Insights
                </Typography>
                
                {emotionResult ? (
                  <Box>
                    <Box sx={{ 
                      display: 'flex', 
                      alignItems: 'center', 
                      gap: 2, 
                      mb: 3,
                      p: 3,
                      borderRadius: 3,
                      background: alpha(getEmotionColor(emotionResult.emotion), 0.1),
                      border: `2px solid ${alpha(getEmotionColor(emotionResult.emotion), 0.3)}`
                    }}>
                      {getEmotionIcon(emotionResult.emotion)}
                      <Box>
                        <Typography variant="h5" sx={{ fontWeight: 700, color: getEmotionColor(emotionResult.emotion) }}>
                          {emotionResult.emotion.charAt(0).toUpperCase() + emotionResult.emotion.slice(1)}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Confidence: {(emotionResult.confidence * 100).toFixed(1)}%
                        </Typography>
                      </Box>
                    </Box>

                    <Divider sx={{ my: 3 }} />
                    
                    <Typography variant="h6" sx={{ mb: 2, fontWeight: 600, color: theme.palette.text.primary }}>
                      Mindful Suggestions:
                    </Typography>
                    <List sx={{ p: 0 }}>
                      {emotionResult.suggestions.map((suggestion, idx) => (
                        <ListItem 
                          key={idx} 
                          sx={{ 
                            py: 1.5,
                            px: 0,
                            '&:hover': {
                              backgroundColor: alpha(theme.palette.primary.main, 0.05),
                              borderRadius: 2,
                              px: 2
                            }
                          }}
                        >
                          <ListItemText 
                            primary={suggestion} 
                            primaryTypographyProps={{ 
                              variant: 'body1',
                              sx: { fontWeight: 500, lineHeight: 1.6 }
                            }}
                          />
                        </ListItem>
                      ))}
                    </List>
                  </Box>
                ) : (
                  <Box sx={{ 
                    textAlign: 'center', 
                    py: 8,
                    color: theme.palette.text.secondary
                  }}>
                    <EmojiEmotions sx={{ fontSize: 64, mb: 2, opacity: 0.5 }} />
                    <Typography variant="h6" sx={{ mb: 1 }}>
                      Ready to explore your emotions?
                    </Typography>
                    <Typography variant="body1">
                      Share your thoughts or upload a photo to begin your mindful journey
                    </Typography>
                  </Box>
                )}
              </Paper>
            </Grid>

            {/* Daily Content Section */}
            <Grid item xs={12}>
              <Box sx={{ mb: 4 }}>
                <Typography 
                  variant="h3" 
                  gutterBottom
                  sx={{ 
                    color: theme.palette.text.primary,
                    mb: 2,
                    fontWeight: 700,
                    textAlign: 'center'
                  }}
                >
                  Daily Inspiration
                </Typography>
                <Typography 
                  variant="h6" 
                  color="text.secondary"
                  sx={{ 
                    textAlign: 'center',
                    maxWidth: '600px',
                    mx: 'auto',
                    mb: 4
                  }}
                >
                  Nurture your mind with thoughtful insights and mindful practices
                </Typography>
              </Box>
              
              <Grid container spacing={3}>
                {dailyContent.map((content, index) => (
                  <Grid item xs={12} md={4} key={index}>
                    <Card 
                      sx={{ 
                        height: '100%',
                        background: isDarkMode 
                          ? 'rgba(26, 32, 44, 0.95)' 
                          : 'rgba(255, 255, 255, 0.95)',
                        backdropFilter: 'blur(20px)',
                        border: `1px solid ${alpha(theme.palette.primary.main, 0.1)}`,
                        borderRadius: 4,
                        transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                        '&:hover': {
                          transform: 'translateY(-8px)',
                          boxShadow: '0 20px 40px rgba(0, 0, 0, 0.12)',
                        },
                      }}
                    >
                      <CardContent sx={{ p: 4 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                          <Avatar
                            sx={{
                              background: 'linear-gradient(135deg, #667eea 0%, #f093fb 100%)',
                              mr: 2
                            }}
                          >
                            <Lightbulb />
                          </Avatar>
                          <Typography variant="h6" sx={{ fontWeight: 600 }}>
                            Daily Wisdom
                          </Typography>
                        </Box>
                        
                        <Typography 
                          variant="body1" 
                          sx={{ 
                            mb: 3,
                            color: theme.palette.text.primary,
                            lineHeight: 1.7,
                            fontSize: '1.05rem'
                          }}
                        >
                          {content.text}
                        </Typography>
                        
                        <Chip 
                          label={content.metadata?.type || 'Inspiration'} 
                          color="primary" 
                          variant="outlined"
                          size="small"
                          sx={{ borderRadius: 20 }}
                        />
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            </Grid>
          </Grid>
        </Container>
      </Box>
    </Box>
  );
}

export default App; 
