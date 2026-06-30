ADMIN_DASHBOARD = """
<!DOCTYPE html>
<html>
<head>
    <title>Admin Portal - Analytics Hub</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { background: #0f172a; color: #f8fafc; font-family: system-ui; padding: 20px; }
        .card { background: rgba(255,255,255,0.05); padding: 20px; border-radius: 12px; margin-bottom: 20px; }
        button { background: #3b82f6; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; }
        select { background: #1e293b; color: white; border: 1px solid #334155; padding: 10px; border-radius: 6px; margin-right: 10px; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #334155; }
        th { background: #1e293b; }
    </style>
</head>
<body>
    <h1>Admin Analytics Hub</h1>
    
    <div style="display: flex; gap: 20px;">
        <!-- Left Column -->
        <div style="flex: 1;">
            <div class="card">
                <h2>1. Upload Student Excel Sheet</h2>
                <p>Columns must be: Name, PRN, Email</p>
                <input type="file" id="excelFile" accept=".xlsx, .xls" style="margin-bottom: 10px;">
                <button onclick="uploadExcel()">Generate Accounts & Download</button>
            </div>
            
            <div class="card">
                <h2>2. Historical Session Reports</h2>
                <select id="timeframeSelect">
                    <option value="daily">Daily Report</option>
                    <option value="weekly">Weekly Report</option>
                    <option value="monthly">Monthly Report</option>
                </select>
                <button onclick="generateReport()">Fetch Report</button>
                
                <table>
                    <thead>
                        <tr><th>Period</th><th>Student</th><th>Avg Engagement</th><th>Avg Frustration</th><th>Avg Boredom</th></tr>
                    </thead>
                    <tbody id="reportTableBody"></tbody>
                </table>
            </div>
        </div>
        
        <!-- Right Column -->
        <div style="flex: 1;">
            <div class="card">
                <h2>3. Live Class Graph (Real-Time)</h2>
                <canvas id="liveChart" height="250"></canvas>
            </div>
        </div>
    </div>

    <script>
        // --- 1. Excel Upload ---
        async function uploadExcel() {
            const file = document.getElementById('excelFile').files[0];
            if (!file) return alert('Select a file first');
            
            const formData = new FormData();
            formData.append('file', file);
            
            const res = await fetch('/api/admin/upload_students', { method: 'POST', body: formData });
            if (res.ok) {
                const blob = await res.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'generated_students.xlsx';
                a.click();
            } else {
                alert('Upload failed');
            }
        }
        
        // --- 2. Historical Reports ---
        async function generateReport() {
            const tf = document.getElementById('timeframeSelect').value;
            const res = await fetch('/api/admin/reports?timeframe=' + tf);
            const data = await res.json();
            
            const tbody = document.getElementById('reportTableBody');
            tbody.innerHTML = '';
            data.reports.forEach(r => {
                tbody.innerHTML += `<tr>
                    <td>${r.period}</td>
                    <td>${r.student_name}</td>
                    <td style="color: #10b981;">${r.avg_engagement.toFixed(2)}</td>
                    <td style="color: #ef4444;">${r.avg_frustration.toFixed(2)}</td>
                    <td style="color: #3b82f6;">${r.avg_boredom.toFixed(2)}</td>
                </tr>`;
            });
        }
        
        // --- 3. Live WebSocket Graph ---
        const ctx = document.getElementById('liveChart').getContext('2d');
        const liveChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'Engagement', borderColor: '#10b981', data: [], tension: 0.4 },
                    { label: 'Boredom', borderColor: '#3b82f6', data: [], tension: 0.4 },
                    { label: 'Frustration', borderColor: '#ef4444', data: [], tension: 0.4 }
                ]
            },
            options: {
                responsive: true,
                scales: { y: { beginAtZero: true, max: 100 } },
                animation: { duration: 0 }
            }
        });
        
        const ws = new WebSocket(`ws://${location.host}/ws/teacher`);
        let timeTick = 0;
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'telemetry' && data.class_average && Object.keys(data.class_average).length > 0) {
                timeTick++;
                liveChart.data.labels.push(timeTick + "s");
                liveChart.data.datasets[0].data.push(data.class_average.engagement);
                liveChart.data.datasets[1].data.push(data.class_average.boredom);
                liveChart.data.datasets[2].data.push(data.class_average.frustration);
                
                // Keep only last 60 points on the graph
                if(liveChart.data.labels.length > 60) {
                    liveChart.data.labels.shift();
                    liveChart.data.datasets.forEach(ds => ds.data.shift());
                }
                liveChart.update();
            }
        };
    </script>
</body>
</html>
"""

TEACHER_DASHBOARD = """
<!DOCTYPE html>
<html>
<head>
    <title>Teacher Portal</title>
    <style>
        body { background: #0f172a; color: #f8fafc; font-family: system-ui; padding: 20px; }
        .card { background: rgba(255,255,255,0.05); padding: 20px; border-radius: 12px; margin-bottom: 20px; }
        button { background: #10b981; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; }
        .danger { background: #ef4444; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 15px; }
        .student-card { background: rgba(0,0,0,0.2); padding: 15px; border-radius: 8px; border: 1px solid #334155; cursor: pointer;}
        #videoModal { display: none; position: fixed; top: 10%; left: 10%; width: 80%; height: 80%; background: #000; border: 2px solid #3b82f6; z-index: 100; text-align: center; }
        #videoModal img { max-width: 100%; max-height: 90%; }
        #alertBanner { display: none; background: #ef4444; color: white; padding: 15px; text-align: center; font-weight: bold; font-size: 20px; border-radius: 8px; margin-bottom: 20px; }
    </style>
</head>
<body>
    <h1>Teacher Portal - Live Class</h1>
    <div id="alertBanner">SLEEPING DETECTED!</div>
    
    <div class="card">
        <button id="startBtn" onclick="startSession()">Start Live Session</button>
        <button class="danger" onclick="endSession()">End Session</button>
        
        <div style="margin-top: 15px; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 8px;">
            <h3 style="margin: 0 0 10px 0; color: #3b82f6;">Class Average (Last 1 Min)</h3>
            <span id="avgBoredom" style="margin-right: 15px;">Boring: 0</span>
            <span id="avgEngagement" style="margin-right: 15px;">Interesting: 0</span>
            <span id="avgFrustration" style="margin-right: 15px;">Frustrated: 0</span>
        </div>
    </div>

    <div class="grid" id="studentGrid"></div>

    <div id="videoModal">
        <button onclick="closeModal()" style="float: right; margin: 10px;" class="danger">Close</button>
        <h2 id="modalTitle">Student Video</h2>
        <img id="liveVideoFeed" src="">
    </div>

    <script>
        let ws;
        let activeVideoUser = null;
        
        function connectWS() {
            ws = new WebSocket(`ws://${location.host}/ws/teacher`);
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if(data.type === 'telemetry') {
                    updateGrid(data.students);
                    if(data.class_average && Object.keys(data.class_average).length > 0) {
                        document.getElementById('avgBoredom').innerText = `Boring: ${data.class_average.boredom}`;
                        document.getElementById('avgEngagement').innerText = `Interesting: ${data.class_average.engagement}`;
                        document.getElementById('avgFrustration').innerText = `Frustrated: ${data.class_average.frustration}`;
                    }
                }
                if(data.type === 'video_frame' && activeVideoUser === data.user_id) {
                    document.getElementById('liveVideoFeed').src = data.frame;
                }
                if(data.type === 'alert') {
                    const banner = document.getElementById('alertBanner');
                    banner.innerText = `SLEEPING DETECTED: ${data.student_name} (>120s)`;
                    banner.style.display = 'block';
                    setTimeout(() => banner.style.display = 'none', 5000);
                }
            };
        }

        function updateGrid(students) {
            const grid = document.getElementById('studentGrid');
            grid.innerHTML = '';
            for (const [id, s] of Object.entries(students)) {
                grid.innerHTML += `
                    <div class="student-card" onclick="viewVideo(${id}, '${s.name}')">
                        <h3>${s.name}</h3>
                        <p>State: ${s.state}</p>
                        <p>Boring: ${s.emotions.boredom || 0}</p>
                        <p>Interesting: ${s.emotions.engagement || 0}</p>
                        <p>Frustrated: ${s.emotions.frustration || 0}</p>
                    </div>`;
            }
        }

        async function startSession() {
            await fetch('/api/teacher/start_session', { method: 'POST' });
            document.getElementById('startBtn').innerText = 'Session Active';
        }
        
        async function endSession() {
            await fetch('/api/teacher/end_session', { method: 'POST' });
            document.getElementById('startBtn').innerText = 'Start Live Session';
        }

        function viewVideo(userId, name) {
            activeVideoUser = userId;
            document.getElementById('modalTitle').innerText = name + "'s Live Feed";
            document.getElementById('videoModal').style.display = 'block';
            ws.send(JSON.stringify({action: 'subscribe_video', user_id: userId}));
        }

        function closeModal() {
            activeVideoUser = null;
            document.getElementById('videoModal').style.display = 'none';
            ws.send(JSON.stringify({action: 'unsubscribe_video'}));
        }

        connectWS();
    </script>
</body>
</html>
"""

USER_DASHBOARD = """
<!DOCTYPE html>
<html>
<head>
    <title>Student Portal</title>
    <style>
        body { background: #0f172a; color: #f8fafc; font-family: system-ui; text-align: center; padding: 50px; }
        video, canvas { border-radius: 12px; margin-top: 20px; border: 2px solid #334155; }
        .hidden { display: none; }
        .lobby { background: rgba(255,255,255,0.05); padding: 40px; border-radius: 12px; font-size: 24px; }
        .active { border-color: #10b981; }
    </style>
</head>
<body>
    <h1>Student Portal</h1>
    
    <div id="lobbyScreen" class="lobby">
        <p>⏳ Waiting for the Teacher to start the session...</p>
    </div>

    <div id="activeScreen" class="hidden">
        <h2 style="color: #10b981;">Session is Live</h2>
        <p>Virtual Camera Pipeline Active. You may now select "Smart Attention Tracker Virtual Camera" in Zoom/GMeet.</p>
        <video id="video" width="640" height="480" autoplay playsinline></video>
        <canvas id="canvas" width="640" height="480" class="hidden"></canvas>
    </div>

    <script>
        let ws;
        let video = document.getElementById('video');
        let canvas = document.getElementById('canvas');
        let ctx = canvas.getContext('2d');
        let streamInterval;

        function connectWS() {
            ws = new WebSocket(`ws://${location.host}/ws/student/student_id_placeholder`);
            ws.onmessage = async (event) => {
                const data = JSON.parse(event.data);
                if (data.action === 'start_session') {
                    document.getElementById('lobbyScreen').classList.add('hidden');
                    document.getElementById('activeScreen').classList.remove('hidden');
                    await startCamera();
                }
                if (data.action === 'end_session') {
                    document.getElementById('activeScreen').classList.add('hidden');
                    document.getElementById('lobbyScreen').classList.remove('hidden');
                    stopCamera();
                }
            };
        }

        async function startCamera() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                video.srcObject = stream;
                streamInterval = setInterval(sendFrame, 1000); // 1 FPS for analytics
            } catch (err) {
                alert("Camera access denied or unavailable.");
            }
        }

        function stopCamera() {
            clearInterval(streamInterval);
            if(video.srcObject) video.srcObject.getTracks().forEach(t => t.stop());
        }

        function sendFrame() {
            if (ws.readyState === WebSocket.OPEN && video.videoWidth) {
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                const frame = canvas.toDataURL('image/jpeg', 0.7);
                ws.send(JSON.stringify({ type: 'frame', image: frame }));
            }
        }

        connectWS();
    </script>
</body>
</html>
"""
