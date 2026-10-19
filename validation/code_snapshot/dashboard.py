"""Localhost live web dashboard for FRC Match Video Downloader & Frame Extractor."""

import argparse
import http.server
import json
import os
import re
import socket
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from progress import MicroProgressTracker


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>FRC Match Video &amp; Frame Pipeline — Live Telemetry</title>
  <style>
    @property --value {
      syntax: '<number>';
      inherits: true;
      initial-value: 0;
    }

    :root {
      --bg-primary: #0b0f19;
      --bg-card: #131b2e;
      --bg-card-alt: #19233c;
      --border-subtle: #263354;
      --text-primary: #f1f5f9;
      --text-secondary: #94a3b8;
      --text-muted: #64748b;
      --accent-blue: #3b82f6;
      --accent-cyan: #06b6d4;
      --accent-green: #10b981;
      --accent-amber: #f59e0b;
      --accent-red: #ef4444;
      --accent-purple: #8b5cf6;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      background-color: var(--bg-primary);
      color: var(--text-primary);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, Helvetica, Arial, sans-serif;
      line-height: 1.5;
      padding: 24px;
      min-height: 100vh;
    }

    .container {
      max-width: 1440px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }

    /* Header */
    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 14px;
      padding: 18px 24px;
    }

    .header-title h1 {
      font-size: 1.35rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .header-title p {
      color: var(--text-secondary);
      font-size: 0.875rem;
      margin-top: 2px;
    }

    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 14px;
      border-radius: 9999px;
      font-size: 0.85rem;
      font-weight: 600;
      background: rgba(59, 130, 246, 0.15);
      color: #60a5fa;
      border: 1px solid rgba(59, 130, 246, 0.35);
    }

    .status-pill.complete {
      background: rgba(16, 185, 129, 0.15);
      color: #34d399;
      border-color: rgba(16, 185, 129, 0.35);
    }

    .pulse-dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: currentColor;
      animation: pulse 1.6s infinite ease-in-out;
    }

    .status-pill.complete .pulse-dot {
      animation: none;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(0.85); }
    }

    /* Top Hero Grid: Progress Ring + Overall Telemetry */
    .hero-grid {
      display: grid;
      grid-template-columns: 240px 1fr;
      gap: 20px;
    }

    @media (max-width: 860px) {
      .hero-grid {
        grid-template-columns: 1fr;
      }
    }

    .card {
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 14px;
      padding: 20px 24px;
    }

    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 14px;
    }

    .card-title {
      font-size: 0.95rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-secondary);
    }

    /* Modern Web Guidance Progress Ring (<progress> + conic-gradient + @property) */
    .ring-card {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      text-align: center;
    }

    .progress-ring-wrapper {
      position: relative;
      display: grid;
      place-items: center;
      margin: 6px 0;
    }

    progress.progress-ring {
      --size: 164px;
      --thickness: 14px;
      --track-color: #1e293b;
      --fill-color: var(--accent-blue);
      --value: 0;
      appearance: none;
      -webkit-appearance: none;
      width: var(--size);
      height: var(--size);
      border-radius: 50%;
      background: conic-gradient(
        var(--fill-color) calc(var(--value) * 1%),
        var(--track-color) 0
      );
      background-clip: border-area;
      border: var(--thickness) solid transparent;
      background-origin: border-box;
      transition: --value 0s ease-in-out;
    }

    @media (prefers-reduced-motion: no-preference) {
      progress.progress-ring {
        transition-duration: 0.25s;
      }
    }

    progress.progress-ring::-webkit-progress-bar,
    progress.progress-ring::-webkit-progress-value {
      display: none;
      background: none;
    }

    progress.progress-ring::-moz-progress-bar {
      display: none;
      background: none;
    }

    @supports not (background-clip: border-area) {
      progress.progress-ring {
        mask-image: radial-gradient(
          transparent calc(50% - var(--thickness)),
          black calc(50% - var(--thickness) + 0.5px)
        );
        -webkit-mask-image: radial-gradient(
          transparent calc(50% - var(--thickness)),
          black calc(50% - var(--thickness) + 0.5px)
        );
        border: 0;
      }
    }

    progress.progress-ring[value="100"] {
      --fill-color: var(--accent-green);
    }

    .progress-ring-content {
      position: absolute;
      display: flex;
      flex-direction: column;
      align-items: center;
    }

    .ring-pct {
      font-size: 1.75rem;
      font-weight: 800;
      font-variant-numeric: tabular-nums;
      color: var(--text-primary);
    }

    .ring-sub {
      font-size: 0.75rem;
      color: var(--text-secondary);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }

    /* Metrics & Linear Bars */
    .telemetry-panel {
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      gap: 18px;
    }

    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
      gap: 12px;
    }

    .kpi-box {
      background: var(--bg-card-alt);
      border: 1px solid var(--border-subtle);
      border-radius: 10px;
      padding: 12px 16px;
    }

    .kpi-label {
      font-size: 0.73rem;
      color: var(--text-secondary);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    .kpi-value {
      font-size: 1.25rem;
      font-weight: 700;
      margin-top: 4px;
      font-variant-numeric: tabular-nums;
    }

    .kpi-sub {
      font-size: 0.74rem;
      color: var(--text-muted);
      margin-top: 2px;
    }

    .bar-section {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    .bar-header {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      font-size: 0.875rem;
    }

    .bar-label {
      font-weight: 600;
      color: var(--text-primary);
    }

    .bar-pct {
      font-weight: 700;
      font-variant-numeric: tabular-nums;
      color: var(--accent-cyan);
    }

    .linear-track {
      width: 100%;
      height: 14px;
      background: #1e293b;
      border-radius: 999px;
      overflow: hidden;
      border: 1px solid var(--border-subtle);
    }

    .linear-fill {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
      border-radius: 999px;
      transition: width 0.2s ease-out;
    }

    .linear-fill.micro {
      background: linear-gradient(90deg, var(--accent-purple), var(--accent-cyan));
    }

    .linear-fill.complete {
      background: var(--accent-green);
    }

    /* Pipeline Stages Roadmap */
    .stages-row {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 10px;
    }

    .stage-step {
      background: var(--bg-card-alt);
      border: 1px solid var(--border-subtle);
      border-radius: 10px;
      padding: 10px 14px;
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 0.84rem;
      color: var(--text-secondary);
    }

    .stage-step.active {
      border-color: var(--accent-blue);
      background: rgba(59, 130, 246, 0.12);
      color: var(--text-primary);
      font-weight: 600;
    }

    .stage-step.done {
      border-color: rgba(16, 185, 129, 0.4);
      background: rgba(16, 185, 129, 0.08);
      color: #34d399;
    }

    .stage-icon {
      width: 22px;
      height: 22px;
      border-radius: 50%;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-size: 0.75rem;
      font-weight: 700;
      background: #1e293b;
      flex-shrink: 0;
    }

    .stage-step.active .stage-icon {
      background: var(--accent-blue);
      color: #fff;
    }

    .stage-step.done .stage-icon {
      background: var(--accent-green);
      color: #0b0f19;
    }

    /* Two-column main details: What's Happening Now vs What's Happening Next */
    .two-col {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
    }

    @media (max-width: 980px) {
      .two-col {
        grid-template-columns: 1fr;
      }
    }

    .now-card-body {
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    .active-item-banner {
      background: var(--bg-card-alt);
      border: 1px solid var(--border-subtle);
      border-left: 4px solid var(--accent-cyan);
      border-radius: 10px;
      padding: 14px 16px;
    }

    .active-item-title {
      font-size: 1.05rem;
      font-weight: 700;
      color: var(--text-primary);
      word-break: break-all;
    }

    .active-item-detail {
      font-size: 0.875rem;
      color: var(--accent-cyan);
      margin-top: 4px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }

    .meta-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
      font-size: 0.85rem;
    }

    @media (max-width: 640px) {
      .meta-grid {
        grid-template-columns: repeat(2, 1fr);
      }
    }

    .meta-item {
      background: rgba(255, 255, 255, 0.02);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 8px 12px;
    }

    .meta-item span {
      display: block;
      font-size: 0.71rem;
      color: var(--text-muted);
      text-transform: uppercase;
    }

    .meta-item strong {
      color: var(--text-primary);
      font-weight: 600;
    }

    /* Tables for Up Next Queue and Recent Completions */
    .table-wrap {
      max-height: 290px;
      overflow-y: auto;
      border: 1px solid var(--border-subtle);
      border-radius: 10px;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.82rem;
      text-align: left;
    }

    thead th {
      position: sticky;
      top: 0;
      background: var(--bg-card-alt);
      color: var(--text-secondary);
      font-weight: 600;
      padding: 10px 12px;
      border-bottom: 1px solid var(--border-subtle);
      z-index: 1;
    }

    tbody td {
      padding: 9px 12px;
      border-bottom: 1px solid rgba(38, 51, 84, 0.5);
      color: var(--text-primary);
    }

    tbody tr:hover {
      background: rgba(255, 255, 255, 0.03);
    }

    .badge {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 6px;
      font-size: 0.72rem;
      font-weight: 600;
    }

    .badge-next {
      background: rgba(245, 158, 11, 0.15);
      color: #fbbf24;
    }

    .badge-done {
      background: rgba(16, 185, 129, 0.15);
      color: #34d399;
    }

    .badge-cached {
      background: rgba(59, 130, 246, 0.15);
      color: #60a5fa;
    }

    .badge-fail {
      background: rgba(239, 68, 68, 0.15);
      color: #f87171;
    }

    .size-pill {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 0.76rem;
      color: #38bdf8;
      font-weight: 600;
      white-space: nowrap;
    }

    /* Log Console */
    .log-console {
      background: #070a12;
      border: 1px solid var(--border-subtle);
      border-radius: 10px;
      padding: 14px;
      height: 220px;
      overflow-y: auto;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 0.8rem;
      color: #cbd5e1;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }

    .log-line {
      white-space: pre-wrap;
      word-break: break-word;
    }

    .log-line.success { color: #34d399; }
    .log-line.warn { color: #fbbf24; }
    .log-line.error { color: #f87171; }
    .log-line.header { color: #60a5fa; font-weight: 600; }

    a {
      color: var(--accent-cyan);
      text-decoration: none;
    }
    a:hover {
      text-decoration: underline;
    }
  </style>
</head>
<body>
  <div class="container">
    <!-- Header -->
    <header class="header">
      <div class="header-title">
        <h1>FRC Match Video Downloader &amp; Frame Extractor</h1>
        <p id="gdrive-path-display">Target Google Drive: Loading...</p>
      </div>
      <div id="status-badge" class="status-pill">
        <span class="pulse-dot"></span>
        <span id="status-text">Running Pipeline</span>
      </div>
    </header>

    <!-- Hero Progress Section -->
    <section class="hero-grid">
      <div class="card ring-card">
        <div class="card-title">Overall Progress</div>
        <div class="progress-ring-wrapper">
          <progress id="overall-ring" value="0" max="100" aria-label="Overall pipeline progress" class="progress-ring"></progress>
          <div class="progress-ring-content">
            <span id="overall-ring-pct" class="ring-pct">0.0%</span>
            <span id="overall-ring-sub" class="ring-sub">Pipeline</span>
          </div>
        </div>
      </div>

      <div class="card telemetry-panel">
        <div class="kpi-grid">
          <div class="kpi-box">
            <div class="kpi-label">Elapsed Time</div>
            <div id="kpi-elapsed" class="kpi-value">00:00</div>
            <div class="kpi-sub">Active runtime</div>
          </div>
          <div class="kpi-box">
            <div class="kpi-label">Estimated Time Left</div>
            <div id="kpi-eta" class="kpi-value">--:--</div>
            <div id="kpi-eta-sub" class="kpi-sub">Based on micro-velocity</div>
          </div>
          <div class="kpi-box">
            <div class="kpi-label">Videos Processed</div>
            <div id="kpi-videos" class="kpi-value">0 / 0</div>
            <div id="kpi-videos-sub" class="kpi-sub">0 downloaded, 0 cached</div>
          </div>
          <div class="kpi-box">
            <div class="kpi-label">Total Disk Size</div>
            <div id="kpi-size" class="kpi-value">0.0 MiB</div>
            <div id="kpi-size-sub" class="kpi-sub">Avg: -- MiB/video</div>
          </div>
          <div class="kpi-box">
            <div class="kpi-label">Frames Extracted</div>
            <div id="kpi-frames" class="kpi-value">0</div>
            <div class="kpi-sub">5 frames per video</div>
          </div>
        </div>

        <div class="bar-section">
          <div class="bar-header">
            <span class="bar-label">Continuous Macro + Micro Pipeline Progress</span>
            <span id="overall-linear-pct" class="bar-pct">0.00%</span>
          </div>
          <div class="linear-track">
            <div id="overall-linear-fill" class="linear-fill"></div>
          </div>
        </div>

        <div class="bar-section">
          <div class="bar-header">
            <span id="micro-bar-label" class="bar-label">Current Task Micro-Progress</span>
            <span id="micro-linear-pct" class="bar-pct">0.0%</span>
          </div>
          <div class="linear-track">
            <div id="micro-linear-fill" class="linear-fill micro"></div>
          </div>
        </div>
      </div>
    </section>

    <!-- Pipeline Stages Roadmap -->
    <section class="card">
      <div class="card-header">
        <span class="card-title">Pipeline Roadmap (Past, Current &amp; Upcoming Stages)</span>
      </div>
      <div id="stages-container" class="stages-row"></div>
    </section>

    <!-- What Is Happening Now & What Is Happening Next -->
    <section class="two-col">
      <!-- Happening Now -->
      <div class="card now-card-body">
        <div class="card-header">
          <span class="card-title">What Is Happening Right Now</span>
          <span id="active-phase-badge" class="badge badge-cached">Initializing</span>
        </div>
        <div class="active-item-banner">
          <div id="active-item-title" class="active-item-title">Waiting for task...</div>
          <div id="active-item-detail" class="active-item-detail">Preparing pipeline...</div>
        </div>
        <div class="meta-grid">
          <div class="meta-item">
            <span>Current Match / Item</span>
            <strong id="meta-match">--</strong>
          </div>
          <div class="meta-item">
            <span>Active File Size</span>
            <strong id="meta-filesize" style="color:#38bdf8">--</strong>
          </div>
          <div class="meta-item">
            <span>Season Year</span>
            <strong id="meta-year">--</strong>
          </div>
          <div class="meta-item" style="grid-column: span 2;">
            <span>Location &amp; Event</span>
            <strong id="meta-location">--</strong>
          </div>
          <div class="meta-item">
            <span>YouTube Source</span>
            <strong id="meta-youtube">--</strong>
          </div>
        </div>
      </div>

      <!-- Happening Next -->
      <div class="card">
        <div class="card-header">
          <span class="card-title">What Is Happening Next (Upcoming Queue)</span>
          <span id="queue-count-badge" class="badge badge-next">0 queued</span>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Dataset</th>
                <th>Year</th>
                <th>Match Key</th>
                <th>Location / Event</th>
                <th>YouTube</th>
              </tr>
            </thead>
            <tbody id="upcoming-tbody">
              <tr><td colspan="6" style="color: var(--text-muted); text-align:center;">Queue will populate as seasons are sampled...</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- Completed Videos & Live Log Console -->
    <section class="two-col">
      <div class="card">
        <div class="card-header">
          <span class="card-title">Recently Processed Videos &amp; File Sizes</span>
          <span id="completed-count-badge" class="badge badge-done">0 finished</span>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Status</th>
                <th>Year</th>
                <th>Match</th>
                <th>Location</th>
                <th>File Size</th>
                <th>Output / Detail</th>
              </tr>
            </thead>
            <tbody id="completed-tbody">
              <tr><td colspan="6" style="color: var(--text-muted); text-align:center;">No videos processed yet.</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <span class="card-title">Live Telemetry &amp; Event Log</span>
        </div>
        <div id="log-console" class="log-console"></div>
      </div>
    </section>
  </div>

  <script>
    const ringEl = document.getElementById('overall-ring');
    const ringPctEl = document.getElementById('overall-ring-pct');
    const overallLinearPct = document.getElementById('overall-linear-pct');
    const overallLinearFill = document.getElementById('overall-linear-fill');
    const microLinearPct = document.getElementById('micro-linear-pct');
    const microLinearFill = document.getElementById('micro-linear-fill');
    const microBarLabel = document.getElementById('micro-bar-label');

    let lastLogCount = 0;

    function escapeHtml(str) {
      if (!str) return '';
      return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    }

    async function pollTelemetry() {
      try {
        const res = await fetch('/api/state', { cache: 'no-store' });
        if (!res.ok) return;
        const data = await res.json();
        renderState(data);
      } catch (e) {
        // Server might be shutting down or restarting
      }
    }

    function renderState(s) {
      // Header
      document.getElementById('gdrive-path-display').textContent =
        'Google Drive Target: ' + (s.gdrive_dir || './downloads/gdrive');

      const statusBadge = document.getElementById('status-badge');
      const statusText = document.getElementById('status-text');
      if (s.is_complete) {
        statusBadge.classList.add('complete');
        statusText.textContent = '100% Complete';
        overallLinearFill.classList.add('complete');
      } else {
        statusBadge.classList.remove('complete');
        statusText.textContent = s.phase_label || 'Running';
        overallLinearFill.classList.remove('complete');
      }

      // Overall Progress Ring & Linear Bar
      const overallPct = Number(s.overall_percent || 0);
      ringEl.setAttribute('value', overallPct.toFixed(2));
      ringEl.style.setProperty('--value', overallPct.toFixed(2));
      ringPctEl.textContent = overallPct.toFixed(1) + '%';
      overallLinearPct.textContent = overallPct.toFixed(2) + '%';
      overallLinearFill.style.width = Math.min(100, Math.max(0, overallPct)) + '%';

      // Micro Bar
      const microPct = Number(s.current_percent || 0);
      microLinearPct.textContent = microPct.toFixed(1) + '%';
      microLinearFill.style.width = Math.min(100, Math.max(0, microPct)) + '%';
      microBarLabel.textContent = (s.phase_label ? '[' + s.phase_label + '] ' : '') + (s.item_label || 'Current Task');

      // KPIs
      document.getElementById('kpi-elapsed').textContent = s.elapsed_str || '00:00';
      document.getElementById('kpi-eta').textContent = s.is_complete ? '00:00' : (s.eta_str || '--:--');
      document.getElementById('kpi-eta-sub').textContent = s.is_complete ? 'Finished!' : 'Based on micro-velocity';
      document.getElementById('kpi-videos').textContent =
        (s.videos_done || 0) + ' / ' + (s.videos_total || 0);
      document.getElementById('kpi-videos-sub').textContent =
        (s.videos_downloaded || 0) + ' downloaded, ' + (s.videos_cached || 0) + ' cached, ' + (s.videos_failed || 0) + ' failed';
      document.getElementById('kpi-size').textContent = s.total_size_str || '0.0 MiB';
      document.getElementById('kpi-size-sub').textContent = 'Avg: ' + (s.avg_size_str || '--');
      document.getElementById('kpi-frames').textContent =
        (s.frames_extracted || 0) + ' / ' + (s.frames_total || 0);

      // Active Task Card
      document.getElementById('active-phase-badge').textContent = s.phase_label || 'Active';
      document.getElementById('active-item-title').textContent = s.item_label || 'Idle';
      document.getElementById('active-item-detail').textContent = s.sub_detail || 'Waiting...';

      const curMeta = s.current_meta || {};
      document.getElementById('meta-match').textContent = curMeta.match_key || s.item_label || '--';
      document.getElementById('meta-filesize').textContent = s.current_file_size || '--';
      document.getElementById('meta-location').textContent =
        curMeta.location ? (curMeta.location + (curMeta.event_name ? ' — ' + curMeta.event_name : '')) : '--';
      document.getElementById('meta-year').textContent = curMeta.year || '--';
      const ytEl = document.getElementById('meta-youtube');
      if (curMeta.youtube_url) {
        ytEl.innerHTML = '<a href="' + escapeHtml(curMeta.youtube_url) + '" target="_blank" rel="noopener">' +
          escapeHtml(curMeta.youtube_id || curMeta.youtube_url) + ' ↗</a>';
      } else {
        ytEl.textContent = '--';
      }

      // Pipeline Roadmap Stages
      const stagesEl = document.getElementById('stages-container');
      if (Array.isArray(s.stages)) {
        stagesEl.innerHTML = s.stages.map((st, idx) => {
          const cls = st.status === 'done' ? 'done' : (st.status === 'active' ? 'active' : '');
          const icon = st.status === 'done' ? '✓' : (idx + 1);
          return '<div class="stage-step ' + cls + '">' +
            '<span class="stage-icon">' + icon + '</span>' +
            '<div><div>' + escapeHtml(st.name) + '</div>' +
            '<div style="font-size:0.72rem;color:var(--text-muted)">' + escapeHtml(st.detail || '') + '</div></div>' +
            '</div>';
        }).join('');
      }

      // Upcoming Queue Table
      const queue = s.upcoming_queue || [];
      document.getElementById('queue-count-badge').textContent = queue.length + ' queued';
      const upTbody = document.getElementById('upcoming-tbody');
      if (queue.length === 0) {
        upTbody.innerHTML = '<tr><td colspan="6" style="color: var(--text-muted); text-align:center;">' +
          (s.is_complete ? 'All queued tasks have been completed!' : 'No items waiting in queue.') + '</td></tr>';
      } else {
        upTbody.innerHTML = queue.slice(0, 40).map((q, i) => (
          '<tr>' +
          '<td><span class="badge badge-next">Next +' + (i + 1) + '</span></td>' +
          '<td>' + escapeHtml(q.dataset || '') + '</td>' +
          '<td>' + escapeHtml(q.year || '') + '</td>' +
          '<td><strong>' + escapeHtml(q.match_key || '') + '</strong></td>' +
          '<td>' + escapeHtml(q.location || '') + (q.event_name ? ' <span style="color:var(--text-muted)">(' + escapeHtml(q.event_name) + ')</span>' : '') + '</td>' +
          '<td><a href="' + escapeHtml(q.youtube_url || '#') + '" target="_blank">' + escapeHtml(q.youtube_id || 'link') + '</a></td>' +
          '</tr>'
        )).join('');
      }

      // Completed Table with File Size Column
      const completed = s.completed_items || [];
      document.getElementById('completed-count-badge').textContent = completed.length + ' finished';
      const compTbody = document.getElementById('completed-tbody');
      if (completed.length === 0) {
        compTbody.innerHTML = '<tr><td colspan="6" style="color: var(--text-muted); text-align:center;">No videos finished yet.</td></tr>';
      } else {
        compTbody.innerHTML = completed.slice(0, 50).map(c => {
          let bClass = 'badge-done';
          if (c.status === 'already_exists' || c.status === 'cached') bClass = 'badge-cached';
          if (c.status === 'failed' || c.status === 'error') bClass = 'badge-fail';
          return '<tr>' +
            '<td><span class="badge ' + bClass + '">' + escapeHtml(c.status) + '</span></td>' +
            '<td>' + escapeHtml(c.year || '') + '</td>' +
            '<td><strong>' + escapeHtml(c.match_key || '') + '</strong></td>' +
            '<td>' + escapeHtml(c.location || '') + '</td>' +
            '<td><span class="size-pill">' + escapeHtml(c.file_size || '--') + '</span></td>' +
            '<td style="font-family:monospace;font-size:0.75rem">' + escapeHtml(c.detail || '') + '</td>' +
            '</tr>';
        }).join('');
      }

      // Live Logs
      const logs = s.logs || [];
      if (logs.length !== lastLogCount) {
        const logConsole = document.getElementById('log-console');
        const wasAtBottom = (logConsole.scrollHeight - logConsole.scrollTop - logConsole.clientHeight) < 40;
        logConsole.innerHTML = logs.map(line => {
          let cls = '';
          if (line.includes('✓') || line.includes('Success')) cls = 'success';
          else if (line.includes('Warning') || line.includes('SKIP')) cls = 'warn';
          else if (line.includes('✗') || line.includes('Error') || line.includes('Failed')) cls = 'error';
          else if (line.includes('===') || line.includes('---')) cls = 'header';
          return '<div class="log-line ' + cls + '">' + escapeHtml(line) + '</div>';
        }).join('');
        if (wasAtBottom || lastLogCount === 0) {
          logConsole.scrollTop = logConsole.scrollHeight;
        }
        lastLogCount = logs.length;
      }
    }

    setInterval(pollTelemetry, 200);
    pollTelemetry();
  </script>
</body>
</html>
"""


def enrich_state_with_disk_sizes(state: Dict[str, Any]) -> Dict[str, Any]:
    """Enriches a telemetry state dict with live file sizes directly from disk without interrupting downloads."""
    gdrive_dir_str = state.get("gdrive_dir") or "./downloads/gdrive"
    gdrive_path = Path(gdrive_dir_str)

    # Map filename -> byte size from disk
    size_by_name: Dict[str, int] = {}
    total_bytes = 0
    video_count_on_disk = 0

    for sub in ("frc_this_year", "frc_multi_year"):
        folder = gdrive_path / sub
        if folder.is_dir():
            for f in folder.iterdir():
                if f.is_file() and not f.name.startswith("."):
                    try:
                        sz = f.stat().st_size
                        size_by_name[f.name] = sz
                        if f.suffix.lower() in {".mp4", ".mkv", ".webm"} and not f.name.endswith((".part", ".ytdl")):
                            total_bytes += sz
                            video_count_on_disk += 1
                    except OSError:
                        pass

    # Enrich completed_items if they don't already have file_size
    for item in state.get("completed_items", []):
        if not item.get("file_size") or item.get("file_size") == "--":
            detail = item.get("detail") or ""
            sz = size_by_name.get(detail, 0)
            if sz > 0:
                item["file_size"] = f"{sz / (1024.0 * 1024.0):.2f} MiB"
                item["file_size_bytes"] = sz

    # Extract active downloading file size from sub_detail (e.g. "45.2% of 14.20MiB @ 9.5MiB/s")
    sub_detail = state.get("sub_detail") or ""
    m = re.search(r"(\d+(?:\.\d+)?)%\s+of\s+(\S+)", sub_detail)
    if m:
        pct = float(m.group(1))
        tot_str = m.group(2)
        num_m = re.match(r"(\d+(?:\.\d+)?)([A-Za-z]+)", tot_str)
        if num_m:
            tot_val = float(num_m.group(1))
            unit = num_m.group(2)
            cur_val = tot_val * (pct / 100.0)
            state["current_file_size"] = f"{cur_val:.2f} / {tot_val:.2f} {unit}"
        else:
            state["current_file_size"] = tot_str
    elif not state.get("current_file_size"):
        # Check if current_meta matches a file on disk
        cur_meta = state.get("current_meta") or {}
        if cur_meta.get("match_key"):
            prefix = f"{cur_meta.get('year')}_{cur_meta.get('event_key')}_{cur_meta.get('match_key')}"
            for fname, sz in size_by_name.items():
                if fname.startswith(prefix):
                    state["current_file_size"] = f"{sz / (1024.0 * 1024.0):.2f} MiB"
                    break

    if total_bytes > 0:
        mib = total_bytes / (1024.0 * 1024.0)
        if mib >= 1024.0:
            state["total_size_str"] = f"{mib / 1024.0:.2f} GiB ({mib:.0f} MiB)"
        else:
            state["total_size_str"] = f"{mib:.2f} MiB"
        avg_mib = mib / max(1, video_count_on_disk)
        state["avg_size_str"] = f"{avg_mib:.2f} MiB/vid ({video_count_on_disk} files)"

    return state


class DashboardServer:
    """Lightweight background HTTP server serving the real-time telemetry webpage."""

    def __init__(
        self,
        tracker: Optional["MicroProgressTracker"] = None,
        host: str = "127.0.0.1",
        port: int = 8765,
        upstream_url: Optional[str] = None,
    ):
        self.tracker = tracker
        self.upstream_url = upstream_url
        self.host = host
        self.port = self._find_available_port(host, port)
        self.httpd: Optional[http.server.ThreadingHTTPServer] = None
        self.thread: Optional[threading.Thread] = None

    @staticmethod
    def _find_available_port(host: str, preferred_port: int) -> int:
        for p in range(preferred_port, preferred_port + 25):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    s.bind((host, p))
                    return p
                except OSError:
                    continue
        return preferred_port

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _get_state_dict(self) -> Dict[str, Any]:
        if self.upstream_url:
            try:
                with urllib.request.urlopen(f"{self.upstream_url.rstrip('/')}/api/state", timeout=2) as resp:
                    raw = json.loads(resp.read().decode("utf-8"))
                    return enrich_state_with_disk_sizes(raw)
            except Exception as e:
                return enrich_state_with_disk_sizes({"phase_label": "Connecting...", "sub_detail": str(e)})
        elif self.tracker:
            return enrich_state_with_disk_sizes(self.tracker.get_state_snapshot())
        return {}

    def start(self, open_browser: bool = False) -> str:
        """Starts the HTTP server on a daemon background thread."""
        server_ref = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path.startswith("/api/state"):
                    payload = json.dumps(server_ref._get_state_dict()).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                else:
                    body = DASHBOARD_HTML.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

            def log_message(self, format, *args):
                pass

        self.httpd = http.server.ThreadingHTTPServer((self.host, self.port), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

        if open_browser:
            try:
                webbrowser.open(self.url)
            except Exception:
                pass

        return self.url

    def stop(self) -> None:
        if self.httpd:
            try:
                self.httpd.shutdown()
                self.httpd.server_close()
            except Exception:
                pass
            self.httpd = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run live companion telemetry dashboard with real-time disk file sizes.")
    parser.add_argument("--upstream", type=str, default="http://127.0.0.1:8765", help="Upstream running dashboard URL")
    parser.add_argument("--port", type=int, default=8766, help="Port to serve enhanced dashboard")
    args = parser.parse_args()

    srv = DashboardServer(upstream_url=args.upstream, port=args.port)
    url = srv.start(open_browser=False)
    print(f"Live File-Size Enhanced Dashboard running at: {url} (proxying {args.upstream})")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        srv.stop()
