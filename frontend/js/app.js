// Configuration
const API_BASE = 'http://127.0.0.1:8000';

// Global State
let currentVideos = [];

// DOM Elements
const navItems = document.querySelectorAll('.nav-item');
const views = document.querySelectorAll('.view-section');

// Upload Elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const progressContainer = document.getElementById('upload-progress-container');
const progressBar = document.getElementById('upload-progress-bar');
const progressText = document.getElementById('upload-percent');
const statusText = document.getElementById('upload-status-text');
const resultMessage = document.getElementById('upload-result');

// Processing Elements
const processingContainer = document.getElementById('processing-container');
const processingStatusText = document.getElementById('processing-status-text');
const processingPercent = document.getElementById('processing-percent');
const processingProgressBar = document.getElementById('processing-progress-bar');
const processingEta = document.getElementById('processing-eta');
const processingFrames = document.getElementById('processing-frames');
const processingEvents = document.getElementById('processing-events');

// Search Elements
const searchBtn = document.getElementById('search-btn');
const searchInput = document.getElementById('search-input');
const thresholdSlider = document.getElementById('threshold-slider');
const thresholdValue = document.getElementById('threshold-value');
const resultsContainer = document.getElementById('search-results-container');
const searchLoading = document.getElementById('search-loading');
const videoSelect = document.getElementById('video-select');

// Analytics Elements
const analyticsVideoSelect = document.getElementById('analytics-video-select');
const loadAnalyticsBtn = document.getElementById('load-analytics-btn');
const analyticsLoading = document.getElementById('analytics-loading');
const analyticsContent = document.getElementById('analytics-content');
const investigationPlayer = document.getElementById('investigation-player');
const videoPlayerContainer = document.getElementById('video-player-container');
const currentEventPanel = document.getElementById('current-event-panel');
const reviewEventType = document.getElementById('review-event-type');
const reviewEventTime = document.getElementById('review-event-time');
const reviewEventScore = document.getElementById('review-event-score');
const reviewEventScoreContainer = document.getElementById('review-event-score-container');
const reviewEventDesc = document.getElementById('review-event-desc');
const statTotalEvents = document.getElementById('stat-total-events');
window.currentReviewingTimestamp = null;
const statNotableEvents = document.getElementById('stat-notable-events');
const overviewText = document.getElementById('overview-text');
const timelineList = document.getElementById('timeline-list');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupNavigation();
    setupUpload();
    setupSearch();
    setupAnalytics();
    fetchVideos();
});

// --- Global Error Panel ---
function showErrorPanel(title, reason, backendResponse = "") {
    console.error(`[ERROR] ${title} | ${reason} | ${backendResponse}`);
    const container = document.getElementById('global-error-panel-container');
    const timestamp = new Date().toISOString().replace('T', ' ').substring(0, 16);
    
    // Auto-dismiss previous ones or stack them. For simplicity, replace.
    const panelHtml = `
        <div class="error-panel">
            <div class="error-panel-header">
                <h3><i class="fa-solid fa-triangle-exclamation"></i> ${title}</h3>
                <button class="error-close-btn" onclick="this.closest('.error-panel').remove()">
                    <i class="fa-solid fa-xmark"></i>
                </button>
            </div>
            <div class="error-panel-body">
                <p><strong>Reason:</strong> ${reason}</p>
                <p><strong>Timestamp:</strong> ${timestamp}</p>
                ${backendResponse ? `<pre>Backend Response:\n${backendResponse}</pre>` : ''}
            </div>
        </div>
    `;
    container.innerHTML = panelHtml;
}

// --- Navigation ---
function setupNavigation() {
    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = item.getAttribute('data-target');
            
            navItems.forEach(nav => nav.classList.remove('active'));
            item.classList.add('active');
            
            views.forEach(view => view.classList.remove('active'));
            document.getElementById(targetId).classList.add('active');
        });
    });
}

// --- Fetch Available Videos ---
async function fetchVideos() {
    try {
        const response = await fetch(`${API_BASE}/videos/`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        const data = await response.json();
        currentVideos = Array.isArray(data) ? data : (data.videos || []);
        
        // Populate select dropdowns
        const optionsHTML = currentVideos.map(v => `<option value="${v.video_id}">${v.original_filename || v.video_id}</option>`).join('');
        videoSelect.innerHTML = `<option value="all">All Videos</option>${optionsHTML}`;
        analyticsVideoSelect.innerHTML = `<option value="all">All Videos</option>${optionsHTML}`;
    } catch (err) {
        showErrorPanel('Fetch Videos Failed', err.message);
    }
}

let selectedFile = null;

// --- Upload Logic ---
function setupUpload() {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', (e) => {
        let dt = e.dataTransfer;
        let files = dt.files;
        if(files.length > 0) selectFile(files[0]);
    }, false);

    fileInput.addEventListener('change', function() {
        if(this.files.length > 0) selectFile(this.files[0]);
    });
    
    document.getElementById('start-upload-btn').addEventListener('click', () => {
        if (selectedFile) handleUpload(selectedFile);
    });
}

function selectFile(file) {
    const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
    const allowed = ['.mp4', '.avi', '.mov'];
    if (!allowed.includes(ext) && !file.type.startsWith('video/')) {
        showUploadResult('Please select a valid video file (.mp4, .avi, .mov).', false);
        selectedFile = null;
        document.getElementById('selected-file-name').textContent = '';
        document.getElementById('start-upload-btn').classList.add('hidden');
        return;
    }
    selectedFile = file;
    document.getElementById('selected-file-name').innerHTML = `<i class="fa-solid fa-check-circle"></i> Selected: ${file.name}`;
    document.getElementById('start-upload-btn').classList.remove('hidden');
    resultMessage.classList.add('hidden');
}

async function handleUpload(file) {
    document.getElementById('start-upload-btn').classList.add('hidden');
    document.getElementById('selected-file-name').textContent = '';
    selectedFile = null;

    const formData = new FormData();
    formData.append('file', file);

    progressContainer.classList.remove('hidden');
    processingContainer.classList.add('hidden');
    resultMessage.classList.add('hidden');
    progressBar.style.width = '10%';
    progressText.textContent = '10%';
    statusText.textContent = 'Uploading...';

    try {
        const response = await fetch(`${API_BASE}/videos/upload`, {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const data = await response.json();
            progressBar.style.width = '100%';
            progressText.textContent = '100%';
            statusText.textContent = 'Upload Complete';
            
            // Mark Upload stage complete
            document.getElementById('stage-upload').className = 'checklist-item completed';
            document.getElementById('stage-upload').innerHTML = '<i class="fa-solid fa-check-circle"></i> Video Uploaded';
            
            // Start polling background job status
            startStatusPolling(data.video_id);
            
            fetchVideos(); // refresh list with new video ID
        } else {
            const errText = await response.text();
            showUploadResult(`Upload failed`, false);
            showErrorPanel('Upload Failed', `HTTP ${response.status}: ${response.statusText}`, errText);
        }
    } catch (err) {
        showUploadResult(`Error: ${err.message}`, false);
        showErrorPanel('Upload Exception', err.message);
    }
}

function showUploadResult(message, isSuccess) {
    resultMessage.textContent = message;
    resultMessage.className = `result-message ${isSuccess ? 'success' : 'error'}`;
    resultMessage.classList.remove('hidden');
    if (!isSuccess) {
        progressContainer.classList.add('hidden');
        processingContainer.classList.add('hidden');
    }
}

let statusInterval = null;

function startStatusPolling(videoId) {
    processingContainer.classList.remove('hidden');
    resultMessage.classList.add('hidden');
    
    // Clear any existing poll
    if (statusInterval) clearInterval(statusInterval);
    
    statusInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE}/videos/${videoId}/status`);
            if (!response.ok) {
                const errText = await response.text();
                showErrorPanel('Status Polling Failed', `HTTP ${response.status}`, errText);
                return;
            }
            const data = await response.json();
            
            processingStatusText.textContent = data.current_step || 'Processing...';
            
            const pct = Math.min(100, Math.max(0, data.progress_percent || 0)).toFixed(1);
            processingPercent.textContent = `${pct}%`;
            processingProgressBar.style.width = `${pct}%`;
            
            processingFrames.innerHTML = `<i class="fa-solid fa-film"></i> Frames: ${data.processed_frames || 0} / ${data.total_frames || 0}`;
            processingEvents.innerHTML = `<i class="fa-solid fa-bolt"></i> Events: ${data.events_generated || 0}`;
            
            if (data.estimated_time_remaining > 0) {
                const mins = Math.floor(data.estimated_time_remaining / 60);
                const secs = Math.floor(data.estimated_time_remaining % 60);
                processingEta.innerHTML = `<i class="fa-regular fa-clock"></i> ETA: ${mins}m ${secs}s`;
            } else {
                processingEta.innerHTML = `<i class="fa-regular fa-clock"></i> ETA: Calculating...`;
            }
            
            // Checklist mapping
            const st = (data.current_step || "").toLowerCase();
            if (st.includes('extracting')) {
                document.getElementById('stage-extract').className = 'checklist-item active';
            } else if (st.includes('analyzing') || st.includes('vlm')) {
                document.getElementById('stage-extract').className = 'checklist-item completed';
                document.getElementById('stage-extract').innerHTML = '<i class="fa-solid fa-check-circle"></i> Frame Extraction';
                document.getElementById('stage-metadata').className = 'checklist-item active';
            } else if (st.includes('aggregating')) {
                document.getElementById('stage-metadata').className = 'checklist-item completed';
                document.getElementById('stage-metadata').innerHTML = '<i class="fa-solid fa-check-circle"></i> Metadata Generation';
                document.getElementById('stage-events').className = 'checklist-item active';
            } else if (st.includes('generating embeddings') || st.includes('indexing')) {
                document.getElementById('stage-events').className = 'checklist-item completed';
                document.getElementById('stage-events').innerHTML = '<i class="fa-solid fa-check-circle"></i> Event Segmentation';
            }
            
            if (data.status === 'complete' || data.status === 'failed') {
                clearInterval(statusInterval);
                statusInterval = null;
                
                if (data.status === 'complete') {
                    processingProgressBar.style.width = '100%';
                    processingPercent.textContent = '100%';
                    showUploadResult(`Ingestion complete! Video is ready for search and analytics.`, true);
                    
                    // Mark all ingestion steps complete
                    ['extract', 'metadata', 'events'].forEach(id => {
                        const el = document.getElementById(`stage-${id}`);
                        if (el) {
                            el.className = 'checklist-item completed';
                            el.innerHTML = `<i class="fa-solid fa-check-circle"></i> ${el.innerText.trim()}`;
                        }
                    });
                    
                    // Narrative and summary are done lazily on the Analytics page, but we mark them complete for UX transparency.
                    // Wait, let's leave them pending, and mark them when Analytics is loaded.
                    
                    // Auto-refresh data and navigate
                    await fetchVideos();
                    
                    // Switch to analytics view after a short delay
                    setTimeout(() => {
                        analyticsVideoSelect.value = videoId;
                        document.querySelector('[data-target="analytics-view"]').click();
                        loadAnalyticsBtn.click();
                    }, 1500);
                } else {
                    showUploadResult(`Processing failed at step: ${data.current_step}`, false);
                    processingProgressBar.style.background = 'var(--danger-color)';
                    showErrorPanel('Processing Failed', data.current_step, JSON.stringify(data, null, 2));
                    
                    // Mark active checklist item as failed
                    document.querySelectorAll('.checklist-item.active').forEach(el => {
                        el.className = 'checklist-item failed';
                        el.innerHTML = `<i class="fa-solid fa-circle-xmark"></i> ${el.innerText.trim()}`;
                    });
                }
            }
        } catch (err) {
            console.error("Status polling failed:", err);
            // Don't show error panel on every polling interval fail to avoid spam
        }
    }, 2000);
}

// --- Search Logic ---
function setupSearch() {
    thresholdSlider.addEventListener('input', (e) => {
        // Show raw percentage value
        thresholdValue.textContent = `${e.target.value}%`;
    });

    searchBtn.addEventListener('click', performSearch);
    searchInput.addEventListener('keypress', (e) => {
        if(e.key === 'Enter') performSearch();
    });
}

async function performSearch() {
    const query = searchInput.value.trim();
    if (!query) return;

    const limit = 12;
    // Convert UI 0‑100 range to 0‑1 for backend
    const threshold = parseFloat(thresholdSlider.value) / 100.0;
    const videoId = videoSelect.value === 'all' ? null : videoSelect.value;

    resultsContainer.innerHTML = '';
    searchLoading.classList.remove('hidden');

    try {
        const reqBody = { query, limit, score_threshold: threshold };
        if (videoId) reqBody.video_ids = [videoId];

        const response = await fetch(`${API_BASE}/api/v1/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(reqBody)
        });

        if (!response.ok) {
            const errText = await response.text();
            throw new Error(`HTTP ${response.status}: ${response.statusText}\n${errText}`);
        }
        const data = await response.json();
        
        if (!data || typeof data !== "object") {
            throw new Error("Invalid server response");
        }
        
        renderSearchResults(data.results || []);
    } catch (err) {
        resultsContainer.innerHTML = `<div class="result-message error">Search Error: ${err.message}</div>`;
        showErrorPanel('Search Failed', err.message);
    } finally {
        searchLoading.classList.add('hidden');
    }
}

function renderSearchResults(results) {
    if (results.length === 0) {
        resultsContainer.innerHTML = `<div style="grid-column: 1/-1; text-align:center; padding: 40px; color: var(--text-secondary);">No events found matching your search.</div>`;
        return;
    }

    const html = results.map(res => {
        const score = (res.score * 100).toFixed(1);
        const event = res;
        const escDesc = event.description.replace(/'/g, "\\'").replace(/"/g, '&quot;');
        
        let thumbnailHtml = "";
        if (event.thumbnail_path) {
            thumbnailHtml = `<img src="${API_BASE}${event.thumbnail_path}" class="result-thumbnail" alt="Event Thumbnail">`;
        }

        return `
            <div class="result-card glass-panel" onclick="playVideoAt('${event.video_id}', toSec('${event.start_time}'), '${event.event_type}', '${escDesc}', '${event.start_time}', '${score}')" style="cursor: pointer;">
                ${thumbnailHtml}
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="result-score"><i class="fa-solid fa-bolt"></i> ${score}% Match</span>
                    <span class="result-time">${event.start_time} - ${event.end_time}</span>
                </div>
                <h4 style="margin-top:8px; color:white;">${formatEventType(event.event_type)}</h4>
                <p class="result-desc">${event.description}</p>
                <div style="margin-top:auto; font-size:12px; color:var(--text-secondary); padding-top:10px; border-top:1px solid var(--panel-border);">
                    Duration: ${event.duration_seconds}s
                </div>
                <button class="btn secondary-btn" style="width:100%; margin-top:15px; pointer-events: none;"><i class="fa-solid fa-play"></i> Review Event</button>
            </div>
        `;
    }).join('');
    
    resultsContainer.innerHTML = html;
}

// --- Analytics Logic ---
function setupAnalytics() {
    loadAnalyticsBtn.addEventListener('click', async () => {
        const videoId = analyticsVideoSelect.value;
        if (!videoId) return;

        // Read optional time filters
        const startAfter = document.getElementById('analytics-start-after').value;
        const endBefore = document.getElementById('analytics-end-before').value;

        analyticsContent.classList.add('hidden');
        analyticsLoading.classList.remove('hidden');

        if (videoId !== 'all') {
            investigationPlayer.src = `${API_BASE}/videos/${videoId}/stream`;
            videoPlayerContainer.classList.remove('hidden');
        } else {
            investigationPlayer.pause();
            investigationPlayer.removeAttribute('src');
            investigationPlayer.load();
            videoPlayerContainer.classList.add('hidden');
        }

        try {
            const videoIds = videoId === 'all' ? currentVideos.map(v => v.video_id) : [videoId];
            const summaries = await Promise.all(videoIds.map(async id => {
                const r = await fetch(`${API_BASE}/api/v1/videos/${id}/summary`);
                if (!r.ok) {
                    const errText = await r.text();
                    throw new Error(`HTTP ${r.status}: ${r.statusText}\n${errText}`);
                }
                return r.json();
            }));
            
            const data = summaries.reduce((acc, curr) => ({
                statistics: { total_events: (acc.statistics?.total_events || 0) + (curr.statistics?.total_events || 0) },
                notable_events: [...(acc.notable_events || []), ...(curr.notable_events || [])],
                timeline: [...(acc.timeline || []), ...(curr.timeline || [])],
                overview: (acc.overview || "") + " " + (curr.overview || ""),
                generation_source: curr.generation_source || acc.generation_source,
                executive_summary: curr.executive_summary || acc.executive_summary,
                incident_narrative: curr.incident_narrative || acc.incident_narrative,
                key_findings: [...(acc.key_findings || []), ...(curr.key_findings || [])],
                recommendations: [...(acc.recommendations || []), ...(curr.recommendations || [])],
                incidents: [...(acc.incidents || []), ...(curr.incidents || [])]
            }), {});

            if (!data || typeof data !== "object") {
                throw new Error("Invalid server response");
            }

            const startSec = startAfter ? toSec(startAfter.split('T')[1]) : null;
            const endSec = endBefore ? toSec(endBefore.split('T')[1]) : null;

            const filterEvents = (events) => {
                if (!startSec && !endSec) return events;
                return events.filter(ev => {
                    const evSec = toSec(ev.timestamp || ev.time_range);
                    if (startSec && evSec < startSec) return false;
                    if (endSec && evSec > endSec) return false;
                    return true;
                });
            };

            const filteredData = {
                ...data,
                notable_events: filterEvents(data.notable_events || []),
                timeline: filterEvents(data.timeline || [])
            };

            renderAnalytics(filteredData);
            
            // Fulfill the narrative/summary UX tracking logic if present on screen
            const stageNarrative = document.getElementById('stage-narrative');
            const stageSummary = document.getElementById('stage-summary');
            if (stageNarrative && stageSummary) {
                stageNarrative.className = 'checklist-item completed';
                stageNarrative.innerHTML = '<i class="fa-solid fa-check-circle"></i> Narrative Reasoning';
                stageSummary.className = 'checklist-item completed';
                stageSummary.innerHTML = '<i class="fa-solid fa-check-circle"></i> Summary Generation';
            }
            
            analyticsLoading.classList.add('hidden');
            analyticsContent.classList.remove('hidden');
        } catch (err) {
            showErrorPanel("Summary Generation Failed", err.message);
            
            // Mark checklist as failed
            const stageNarrative = document.getElementById('stage-narrative');
            const stageSummary = document.getElementById('stage-summary');
            if (stageNarrative && stageSummary) {
                stageNarrative.className = 'checklist-item failed';
                stageNarrative.innerHTML = '<i class="fa-solid fa-circle-xmark"></i> Narrative Reasoning Failed';
                stageSummary.className = 'checklist-item failed';
                stageSummary.innerHTML = '<i class="fa-solid fa-circle-xmark"></i> Summary Generation Failed';
            }
            
            analyticsLoading.classList.add('hidden');
            
            // Render blank safe state
            renderAnalytics({
                statistics: { total_events: 0 },
                notable_events: [],
                timeline: [],
                overview: "Summary unavailable due to backend failure.",
                generation_source: "Error"
            });
            analyticsContent.classList.remove('hidden');
        }
    });
}

function renderAnalytics(data) {
    if (!data || typeof data !== "object") {
        showErrorPanel("Validation Error", "Invalid data payload in renderAnalytics");
        return;
    }

    statTotalEvents.textContent = data.statistics?.total_events || 0;
    statNotableEvents.textContent = (data.notable_events?.length || 0);
    
    const overviewStr = (data.overview && data.overview.trim().length > 0) ? data.overview : "No significant incidents detected.";
    overviewText.textContent = overviewStr;

    // Render Status Badge
    const badgeContainer = document.getElementById('ai-status-badge-container');
    if (badgeContainer) {
        const source = data.generation_source || "Unknown";
        let badgeHtml = "";
        
        if (source.includes("Gemini")) {
            badgeHtml = `<span class="status-badge ai-active"><i class="fa-solid fa-circle-check"></i> AI Reasoning Active</span>
                         <span style="font-size: 11px; color: var(--text-secondary); margin-left: 8px;">Generated By: ${source}</span>`;
        } else if (source.includes("Legacy")) {
            badgeHtml = `<span class="status-badge legacy-fallback"><i class="fa-solid fa-triangle-exclamation"></i> Legacy Fallback Active</span>
                         <span style="font-size: 11px; color: var(--text-secondary); margin-left: 8px;">Generated By: ${source}</span>`;
        } else if (source === "Error") {
            badgeHtml = `<span class="status-badge error-state"><i class="fa-solid fa-circle-xmark"></i> Generation Failed</span>`;
        } else {
            badgeHtml = `<span class="status-badge legacy-fallback"><i class="fa-solid fa-circle-info"></i> ${source}</span>`;
        }
        badgeContainer.innerHTML = badgeHtml;
    }

    const eventsToRender = (data.notable_events && data.notable_events.length > 0) ? data.notable_events : data.timeline;

    if (eventsToRender && eventsToRender.length > 0) {
        timelineList.innerHTML = eventsToRender.map(ev => {
            const sevClass = ev.severity === 'high' ? 'high-severity' : (ev.severity === 'medium' ? 'medium-severity' : '');
            const tagsHtml = (ev.tags || []).map(t => `<span class="tag">${t}</span>`).join('');
            const evDesc = ev.description || ev.summary || '';
            const escDesc = evDesc.replace(/'/g, "\\'").replace(/"/g, '&quot;');
            const timeStr = ev.timestamp || ev.time_range;
            const timeSec = toSec(timeStr);
            
            let posterHtml = '';
            if (ev.poster_frame) {
                posterHtml = `<img src="${API_BASE}${ev.poster_frame}" class="timeline-poster" alt="Event Poster" onerror="this.style.display='none'">`;
            } else if (ev.thumbnail_path) {
                posterHtml = `<img src="${API_BASE}${ev.thumbnail_path}" class="timeline-poster" alt="Event Poster" onerror="this.style.display='none'">`;
            }
            
            return `
                <li class="timeline-item ${sevClass}" data-time-sec="${timeSec}" onclick="playVideoAt('${ev.video_id}', ${timeSec}, '${ev.event_type}', '${escDesc}', '${timeStr}', null)" style="cursor: pointer;">
                    <div class="timeline-time">${timeStr}</div>
                    <div class="timeline-content">
                        <div class="timeline-title">
                            <span>${formatEventType(ev.event_type)}</span>
                            <span style="color: ${ev.severity==='high' ? 'var(--danger-color)' : (ev.severity==='medium' ? 'var(--warning-color)' : 'var(--text-secondary)')}; font-size: 12px; text-transform: uppercase;">
                                ${ev.severity || 'INFO'}
                            </span>
                        </div>
                        ${posterHtml}
                        <div class="timeline-desc" style="margin-top: 12px;">${ev.description}</div>
                        ${ev.reason ? `<div style="font-size: 13px; color: var(--accent-color); margin-top:8px;">Reason: ${ev.reason}</div>` : ''}
                        <div class="tags">${tagsHtml}</div>
                    </div>
                </li>
            `;
        }).join('');

        // Apply active state if applicable
        if (window.currentReviewingTimestamp !== null) {
            document.querySelectorAll('.timeline-item').forEach(el => {
                if (el.dataset.timeSec && Math.abs(parseFloat(el.dataset.timeSec) - window.currentReviewingTimestamp) < 1.0) {
                    el.classList.add('active');
                }
            });
        }
    } else {
        timelineList.innerHTML = '<li style="color: var(--text-secondary);">No events recorded in timeline.</li>';
    }

    // Render Detailed Report
    const reportContainer = document.getElementById('detailed-report-container');
    if (data.executive_summary || data.incident_narrative) {
        reportContainer.classList.remove('hidden');
        if (data.executive_summary) {
            overviewText.textContent = data.executive_summary;
        }
        document.getElementById('report-narrative').textContent = data.incident_narrative || "No narrative available.";
        
        const findingsList = document.getElementById('report-findings');
        findingsList.innerHTML = (data.key_findings || []).map(f => `<li style="margin-bottom: 6px;">${f}</li>`).join('');
        
        const recsList = document.getElementById('report-recommendations');
        recsList.innerHTML = (data.recommendations || []).map(r => `<li style="margin-bottom: 6px;">${r}</li>`).join('');
    } else {
        reportContainer.classList.add('hidden');
    }

    // Render Incident Cards
    const incidentsContainer = document.getElementById('incidents-container');
    const incidentsGrid = document.getElementById('incidents-grid');
    
    if (data.incidents && data.incidents.length > 0) {
        incidentsContainer.classList.remove('hidden');
        incidentsGrid.innerHTML = data.incidents.map(inc => {
            const timeStr = inc.chain_events && inc.chain_events.length > 0 ? inc.chain_events[0].start_time : '';
            const timeSec = toSec(timeStr);
            const incDesc = inc.summary || inc.description || '';
            const escDesc = incDesc.replace(/'/g, "\\'").replace(/"/g, '&quot;');
            const incType = inc.primary_incident_type || inc.incident_type || '';
            
            let heroHtml = "";
            if (inc.poster_frame) {
                heroHtml = `<img src="${API_BASE}${inc.poster_frame}" class="incident-hero" alt="Incident Hero Image" onerror="this.style.display='none'">`;
            } else if (inc.thumbnail_path) {
                heroHtml = `<img src="${API_BASE}${inc.thumbnail_path}" class="incident-hero" alt="Incident Hero Image" onerror="this.style.display='none'">`;
            }

            return `
                <div class="result-card glass-panel" onclick="playVideoAt('${data.video_id || document.getElementById('analytics-video-select').value}', ${timeSec}, '${incType}', '${escDesc}', '${timeStr}', null)" style="cursor: pointer;">
                    ${heroHtml}
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span class="result-score" style="background: rgba(239, 68, 68, 0.1); color: var(--danger-color);"><i class="fa-solid fa-triangle-exclamation"></i> ${(inc.severity || '').toUpperCase()}</span>
                    </div>
                    <h4 style="margin-top:12px; color:white; font-size: 16px;">${formatEventType(incType)}</h4>
                    <p class="result-desc" style="margin-top: 8px;">${incDesc}</p>
                    <div style="margin-top:auto; font-size:12px; color:var(--text-secondary); padding-top:12px; border-top:1px solid var(--panel-border);">
                        ${inc.chain_events ? inc.chain_events.length : 0} Correlated Events
                    </div>
                </div>
            `;
        }).join('');
    } else {
        incidentsContainer.classList.add('hidden');
        incidentsGrid.innerHTML = '';
    }
}

// Utils
function formatEventType(typeStr) {
    return (typeStr || "unknown_event").split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
}

// Helper to convert HH:MM:SS or HH:MM to seconds
function toSec(t) {
    if (!t) return 0;
    const parts = t.toString().split(':').map(Number);
    if (parts.length === 2) return parts[0] * 3600 + parts[1] * 60;
    if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
    return 0;
}

// Global player navigation function
function playVideoAt(videoId, seconds, type = null, desc = null, time = null, score = null) {
    if (!videoId) return;
    console.log(`[playVideoAt] videoId=${videoId}, seconds=${seconds}`);
    
    window.currentReviewingTimestamp = seconds;
    
    // Highlight matching timeline items immediately if they exist
    document.querySelectorAll('.timeline-item').forEach(el => {
        el.classList.remove('active');
        if (el.dataset.timeSec && Math.abs(parseFloat(el.dataset.timeSec) - seconds) < 1.0) {
            el.classList.add('active');
        }
    });

    // Populate Current Event Panel
    if (type) {
        currentEventPanel.classList.remove('hidden');
        reviewEventType.textContent = formatEventType(type);
        reviewEventTime.textContent = time || 'Unknown';
        reviewEventDesc.textContent = desc || 'No description available.';
        if (score) {
            reviewEventScoreContainer.style.display = 'block';
            reviewEventScore.textContent = score + '%';
        } else {
            reviewEventScoreContainer.style.display = 'none';
        }
    } else {
        currentEventPanel.classList.add('hidden');
    }
    
    // Navigate to analytics view if not already there
    const analyticsViewBtn = document.querySelector('.nav-item[data-target="analytics-view"]');
    if (!analyticsViewBtn.classList.contains('active')) {
        analyticsViewBtn.click();
    }
    
    const attemptSeekAndPlay = () => {
        if (investigationPlayer.readyState >= 1) { // HAVE_METADATA
            investigationPlayer.currentTime = seconds;
            const playPromise = investigationPlayer.play();
            if (playPromise !== undefined) {
                playPromise.then(_ => console.log("[playVideoAt] Playback started.")).catch(e => console.error("Auto-play prevented", e));
            }
            investigationPlayer.scrollIntoView({ behavior: 'smooth', block: 'center' });
        } else {
            investigationPlayer.addEventListener('loadedmetadata', attemptSeekAndPlay, { once: true });
        }
    };
    
    // Check if the video is already loaded
    if (analyticsVideoSelect.value === videoId && !videoPlayerContainer.classList.contains('hidden')) {
        attemptSeekAndPlay();
    } else {
        // Load the video via the analytics workflow
        analyticsVideoSelect.value = videoId;
        loadAnalyticsBtn.click();
        attemptSeekAndPlay();
    }
}
