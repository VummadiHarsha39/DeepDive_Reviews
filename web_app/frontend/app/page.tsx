'use client'; // This is a Client Component

import { useState } from 'react';
import { Bar } from 'react-chartjs-2';
// Import necessary Chart.js components
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { FaRegSmile, FaRegMeh, FaRegFrown, FaSpinner } from 'react-icons/fa'; // Icons for verdicts
import { FiUpload, FiDownload } from 'react-icons/fi'; // Icons for file operations
import Papa from 'papaparse'; // For CSV parsing/unparsing

// Register Chart.js components (important for them to work)
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

// --- Interfaces for API Response Data ---
interface EmotionData {
  [key: string]: number; // e.g., { "joy": 0.8, "anger": 0.1 }
}

interface AnalysisResult {
  verdict: 'real' | 'fake' | 'unknown';
  confidence: number;
  emotions: EmotionData;
  explanation: string;
  customer_intent_summary: string;
}

// --- API Base URL Configuration ---
// Make sure this matches where your FastAPI backend is running
// For local testing: 'http://localhost:8000'
// For deployment: 'https://your-deployed-backend-url.cloudrun.app' (Not used for local-only)
const API_BASE_URL = 'http://localhost:8000'; // Hardcoded for local-only operation

// --- Main Page Component ---
export default function Home() {
  const [reviewText, setReviewText] = useState<string>('');
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [batchResults, setBatchResults] = useState<AnalysisResult[] | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // --- Handlers for Single Review Analysis ---
  const handleAnalyzeSingleReview = async () => {
    setLoading(true);
    setAnalysisResult(null); // Clear previous single analysis result
    setBatchResults(null); // Clear previous batch results
    setError(null); // Clear previous errors

    try {
      const response = await fetch(`${API_BASE_URL}/analyze_review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ review_text: reviewText }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`API Error ${response.status}: ${errorData.detail || response.statusText}`);
      }

      const data: AnalysisResult = await response.json();
      setAnalysisResult(data);
    } catch (err: any) {
      console.error('Error analyzing single review:', err);
      setError(err.message || 'An unknown error occurred during analysis.');
    } finally {
      setLoading(false);
    }
  };

  // --- Handlers for Batch Review Analysis (CSV Upload) ---
  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setLoading(true);
      setError(null);
      setAnalysisResult(null); // Clear previous single analysis result
      setBatchResults(null); // Clear previous batch results

      Papa.parse(file, {
        header: true, // Assuming CSV has a header row like 'review_text'
        skipEmptyLines: true,
        complete: async (results) => {
          // Filter out rows that don't have a 'review_text' property
          const reviews = results.data.map((row: any) => row.review_text).filter(Boolean); 

          if (reviews.length === 0) {
            setError("No 'review_text' column found in CSV or file is empty.");
            setLoading(false);
            return;
          }

          try {
            const response = await fetch(`${API_BASE_URL}/batch_analyze_reviews`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ reviews }),
            });

            if (!response.ok) {
              const errorData = await response.json();
              throw new Error(`API Error ${response.status}: ${errorData.detail || response.statusText}`);
            }

            const data: { results: AnalysisResult[] } = await response.json();
            // Store the original review text with the result for CSV download
            const resultsWithOriginalText = data.results.map((r: any, idx: number) => ({
                ...r,
                original_text: reviews[idx] // Attach original text
            }));
            setBatchResults(resultsWithOriginalText);
          } catch (err: any) {
            console.error('Error analyzing batch reviews:', err);
            setError(err.message || 'An unknown error occurred for batch analysis.');
          } finally {
            setLoading(false);
          }
        },
        error: (err: any) => {
          setError(`CSV Parsing Error: ${err.message}`);
          setLoading(false);
        }
      });
    }
  };

  // --- Handler for CSV Download of Batch Results ---
  const handleDownloadCSV = () => {
    if (batchResults) {
      // Prepare data for CSV, flattening emotions
      const csvData = batchResults.map(result => {
        const flattenedEmotions = Object.entries(result.emotions).reduce((acc, [key, value]) => {
          acc[`Emotion_${key}`] = value;
          return acc;
        }, {} as Record<string, number>); // Use Record for type safety

        return {
          "Original Review Text": (result as any).original_text || '', // Get attached original text
          "Verdict": result.verdict,
          "Confidence": result.confidence,
          "Explanation": result.explanation,
          "Customer Intent Summary": result.customer_intent_summary,
          ...flattenedEmotions, // Spread the flattened emotions
        };
      });

      const csv = Papa.unparse(csvData, { header: true }); // Ensure header row
      const blob = new Blob([csv], { type: 'text/csv;charset=utf-utf-8;' });
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.setAttribute('download', 'review_analysis_results.csv');
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  // --- Helper Functions for UI ---
  const getVerdictIcon = (verdict: string) => {
    if (verdict === 'fake') return <FaRegFrown className="text-red-400 text-3xl" />; // Changed color
    if (verdict === 'real') return <FaRegSmile className="text-green-400 text-3xl" />; // Changed color
    return <FaRegMeh className="text-gray-400 text-3xl" />;
  };

  const getEmotionChartData = (emotions: EmotionData) => {
    // Filter out very low probabilities for cleaner chart
    const filteredEmotions = Object.entries(emotions).filter(([, value]) => value > 0.01);
    const labels = filteredEmotions.map(([key]) => key);
    const data = filteredEmotions.map(([, value]) => value);

    return {
      labels: labels,
      datasets: [
        {
          label: 'Emotion Intensity',
          data: data,
          backgroundColor: 'rgba(59, 130, 246, 0.6)', // Bright blue for bars
          borderColor: 'rgba(59, 130, 246, 1)',
          borderWidth: 1,
        },
      ],
    };
  };

  // --- Main Render Function (JSX) ---
  return (
    <div className="min-h-screen bg-gray-900 p-8 font-sans text-gray-100"> {/* Dark background, light text */}
      <h1 className="text-4xl font-bold text-center mb-10 text-blue-400">Emotional Counterfeit Detector</h1> {/* Blue title */}

      <div className="max-w-4xl mx-auto bg-gray-800 p-8 rounded-lg shadow-2xl shadow-blue-500/20 border border-blue-700"> {/* Dark card with subtle blue glow */}
        <h2 className="text-2xl font-semibold mb-6 text-blue-400">Analyze a Single Review</h2> {/* Blue subtitle */}
        <textarea
          className="w-full p-4 border border-gray-700 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-400 mb-4 bg-gray-700 text-gray-100 placeholder-gray-400" // Dark input
          rows={6}
          placeholder="Paste your review here..."
          value={reviewText}
          onChange={(e) => setReviewText(e.target.value)}
          disabled={loading}
        ></textarea>
        <button
          onClick={handleAnalyzeSingleReview}
          className="w-full bg-blue-500 text-white py-3 px-6 rounded-md hover:bg-blue-600 transition duration-300 flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed"
          disabled={loading || !reviewText.trim()}
        >
          {loading && <FaSpinner className="animate-spin mr-2" />}
          Analyze Review
        </button>

        <div className="mt-8 border-t border-gray-700 pt-8"> {/* Darker separator */}
          <h2 className="text-2xl font-semibold mb-6 text-blue-400">Batch Analysis (Upload CSV)</h2> {/* Blue subtitle */}
          <div className="flex items-center justify-center w-full">
            <label htmlFor="csv-upload" className="flex flex-col items-center justify-center w-full h-32 border-2 border-blue-700 border-dashed rounded-lg cursor-pointer bg-gray-700 hover:bg-gray-600 transition duration-300 text-gray-300"> {/* Darker upload area */}
              <div className="flex flex-col items-center justify-center pt-5 pb-6">
                <FiUpload className="text-blue-400 text-4xl mb-2" /> {/* Blue icon */}
                <p className="mb-2 text-sm"><span className="font-semibold">Click to upload</span> or drag and drop</p>
                <p className="text-xs">CSV file with a 'review_text' column</p>
              </div>
              <input id="csv-upload" type="file" className="hidden" accept=".csv" onChange={handleFileUpload} disabled={loading} />
            </label>
          </div>
        </div>

        {error && (
          <div className="bg-red-900/30 border border-red-700 text-red-300 px-4 py-3 rounded relative mt-6" role="alert"> {/* Darker error message */}
            <strong className="font-bold">Error!</strong>
            <span className="block sm:inline"> {error}</span>
          </div>
        )}

        {loading && (
          <div className="text-center mt-6">
            <FaSpinner className="animate-spin text-blue-400 text-4xl mx-auto" /> {/* Blue spinner */}
            <p className="text-gray-400 mt-2">Analyzing...</p>
          </div>
        )}

        {/* Display for Single Analysis Result */}
        {analysisResult && !loading && (
          <div className="mt-8 p-6 bg-blue-900/30 rounded-lg shadow-inner shadow-blue-500/10 border border-blue-700"> {/* Dark result card with subtle inner glow */}
            <h3 className="text-xl font-bold mb-4 flex items-center gap-3 text-blue-300"> {/* Lighter blue for verdict text */}
              {getVerdictIcon(analysisResult.verdict)}
              Verdict: <span className={analysisResult.verdict === 'fake' ? 'text-red-400' : 'text-green-400'}>{analysisResult.verdict.toUpperCase()}</span>
              <span className="text-gray-400 ml-2 text-base">({(analysisResult.confidence * 100).toFixed(1)}% Confidence)</span>
            </h3>
            <p className="mb-4 text-gray-200 font-medium">**Explanation:** <span className="font-normal">{analysisResult.explanation}</span></p>
            <p className="mb-4 text-gray-200 font-medium">**Customer Intent:** <span className="font-normal">{analysisResult.customer_intent_summary}</span></p>

            {Object.keys(analysisResult.emotions).length > 0 && (
                <div className="mt-6">
                <h4 className="text-lg font-semibold mb-3 text-blue-300">Emotion Profile:</h4> {/* Lighter blue subtitle */}
                <div className="h-64">
                    <Bar
                    data={getEmotionChartData(analysisResult.emotions)}
                    options={{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                        legend: { display: false },
                        title: { display: false },
                        },
                        scales: {
                            y: { 
                                beginAtZero: true, 
                                max: 1, 
                                grid: { color: 'rgba(255,255,255,0.1)' }, // Lighter grid lines
                                ticks: { color: 'rgb(156, 163, 175)' } // Gray ticks
                            },
                            x: { 
                                grid: { color: 'rgba(255,255,255,0.1)' }, // Lighter grid lines
                                ticks: { color: 'rgb(156, 163, 175)' } // Gray ticks
                            },
                        },
                    }}
                    />
                </div>
                </div>
            )}
          </div>
        )}

        {/* Display for Batch Analysis Results */}
        {batchResults && !loading && (
          <div className="mt-8 p-6 bg-purple-900/30 rounded-lg shadow-inner shadow-purple-500/10 border border-purple-700"> {/* Dark batch result card with subtle inner glow */}
            <h3 className="text-xl font-bold mb-4 text-purple-300">Batch Analysis Results ({batchResults.length} reviews)</h3> {/* Lighter purple title */}
            <button
              onClick={handleDownloadCSV}
              className="mb-4 bg-green-500 text-white py-2 px-4 rounded-md hover:bg-green-600 transition duration-300 flex items-center disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={!batchResults || batchResults.length === 0}
            >
              <FiDownload className="mr-2" /> Download CSV
            </button>
            <div className="overflow-x-auto">
              <table className="min-w-full bg-gray-800 border border-gray-700 text-sm text-gray-100"> {/* Darker table */}
                <thead>
                  <tr>
                    <th className="py-2 px-4 border-b border-gray-700 text-left text-gray-400">Review Text</th> {/* Gray header text */}
                    <th className="py-2 px-4 border-b border-gray-700 text-left text-gray-400">Verdict</th>
                    <th className="py-2 px-4 border-b border-gray-700 text-left text-gray-400">Confidence</th>
                    <th className="py-2 px-4 border-b border-gray-700 text-left text-gray-400">Explanation</th>
                    <th className="py-2 px-4 border-b border-gray-700 text-left text-gray-400">Customer Intent</th>
                  </tr>
                </thead>
                <tbody>
                  {batchResults.map((result, index) => (
                    <tr key={index} className={index % 2 === 0 ? 'bg-gray-700' : 'bg-gray-800'}> {/* Alternating dark rows */}
                      <td className="py-2 px-4 border-b border-gray-700 max-w-xs overflow-hidden text-ellipsis whitespace-nowrap">{ (result as any).original_text || 'Text not available' }</td>
                      <td className="py-2 px-4 border-b border-gray-700 font-semibold" style={{ color: result.verdict === 'fake' ? '#F87171' : '#4ADE80' }}>{result.verdict.toUpperCase()}</td> {/* Adjusted bright colors */}
                      <td className="py-2 px-4 border-b border-gray-700">{(result.confidence * 100).toFixed(1)}%</td>
                      <td className="py-2 px-4 border-b border-gray-700 text-sm max-w-md overflow-hidden text-ellipsis whitespace-nowrap">{result.explanation}</td>
                      <td className="py-2 px-4 border-b border-gray-700 text-sm max-w-sm overflow-hidden text-ellipsis whitespace-nowrap">{result.customer_intent_summary}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Feedback / Contact Section */}
        <div className="mt-12 text-center text-gray-400">
            <p>For feedback, please contact me through my website:</p>
            <p className="mt-2">
                <a href="https://vummadiharsha39.github.io/" target="_blank" rel="noopener noreferrer" 
                   className="text-blue-400 hover:text-blue-300 underline transition duration-300">
                    https://vummadiharsha39.github.io/
                </a>
            </p>
        </div>
      </div>
    </div>
  );
}