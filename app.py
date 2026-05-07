"""
Bioinformatics Sequence Alignment Visualizer
A comprehensive tool for visualizing sequence alignment algorithms
"""

from flask import Flask, render_template, request, jsonify, session
import json
import numpy as np
from datetime import datetime
import traceback as tb

app = Flask(__name__)
app.secret_key = 'bioinformatics_aligner_secret_key_2024'

class SequenceAligner:
    """Core alignment algorithms implementation"""
    
    def __init__(self, seq1, seq2, match_score, mismatch_penalty, gap_penalty, algorithm):
        self.seq1 = seq1.upper()
        self.seq2 = seq2.upper()
        self.match_score = match_score
        self.mismatch_penalty = mismatch_penalty
        self.gap_penalty = gap_penalty
        self.algorithm = algorithm
        self.matrix = None
        self.traceback_matrix = None
        self.aligned_seq1 = ""
        self.aligned_seq2 = ""
        self.alignment_score = 0
        self.steps = []
        
    def score_function(self, a, b):
        """Calculate score for character pair"""
        if a == b:
            return self.match_score
        else:
            return self.mismatch_penalty
    
    def create_matrices(self):
        """Initialize scoring and traceback matrices"""
        rows = len(self.seq1) + 1
        cols = len(self.seq2) + 1
        
        self.matrix = [[0] * cols for _ in range(rows)]
        self.traceback_matrix = [[''] * cols for _ in range(rows)]
        
        # Initialize first row and column with gaps
        if self.algorithm == "needleman_wunsch":
            for i in range(rows):
                self.matrix[i][0] = i * self.gap_penalty
                self.traceback_matrix[i][0] = 'up'
            for j in range(cols):
                self.matrix[0][j] = j * self.gap_penalty
                self.traceback_matrix[0][j] = 'left'
        
        self.traceback_matrix[0][0] = 'start'
        
        return self.matrix, self.traceback_matrix
    
    def fill_matrix(self):
        """Fill the scoring matrix with step-by-step recording"""
        rows, cols = len(self.seq1) + 1, len(self.seq2) + 1
        
        for i in range(1, rows):
            for j in range(1, cols):
                # Calculate three possible scores
                diagonal = self.matrix[i-1][j-1] + self.score_function(self.seq1[i-1], self.seq2[j-1])
                up = self.matrix[i-1][j] + self.gap_penalty
                left = self.matrix[i][j-1] + self.gap_penalty
                
                # Choose maximum (or 0 for Smith-Waterman)
                if self.algorithm == "smith_waterman":
                    scores = [0, diagonal, up, left]
                    max_score = max(scores)
                    
                    if max_score == 0:
                        direction = 'stop'
                    elif max_score == diagonal:
                        direction = 'diag'
                    elif max_score == up:
                        direction = 'up'
                    else:
                        direction = 'left'
                else:  # Needleman-Wunsch
                    scores = [diagonal, up, left]
                    max_score = max(scores)
                    
                    if max_score == diagonal:
                        direction = 'diag'
                    elif max_score == up:
                        direction = 'up'
                    else:
                        direction = 'left'
                
                self.matrix[i][j] = max_score
                self.traceback_matrix[i][j] = direction
                
                # Record step for visualization
                self.steps.append({
                    'i': i,
                    'j': j,
                    'seq1_char': self.seq1[i-1],
                    'seq2_char': self.seq2[j-1],
                    'diagonal_score': diagonal,
                    'up_score': up,
                    'left_score': left,
                    'chosen_score': max_score,
                    'chosen_direction': direction,
                    'matrix_snapshot': [row[:] for row in self.matrix]
                })
        
        return self.matrix, self.steps
    
    def traceback(self):
        """Perform traceback to find optimal alignment"""
        aligned_seq1 = []
        aligned_seq2 = []
        
        if self.algorithm == "needleman_wunsch":
            i, j = len(self.seq1), len(self.seq2)
        else:  # Smith-Waterman
            # Find maximum score in matrix
            max_score = -float('inf')
            max_i, max_j = 0, 0
            for i in range(len(self.matrix)):
                for j in range(len(self.matrix[0])):
                    if self.matrix[i][j] > max_score:
                        max_score = self.matrix[i][j]
                        max_i, max_j = i, j
            i, j = max_i, max_j
            self.alignment_score = max_score
        
        trace_path = []
        
        while True:
            trace_path.append((i, j))
            direction = self.traceback_matrix[i][j]
            
            if direction == 'diag':
                aligned_seq1.append(self.seq1[i-1] if i > 0 else '-')
                aligned_seq2.append(self.seq2[j-1] if j > 0 else '-')
                i -= 1
                j -= 1
            elif direction == 'up':
                aligned_seq1.append(self.seq1[i-1] if i > 0 else '-')
                aligned_seq2.append('-')
                i -= 1
            elif direction == 'left':
                aligned_seq1.append('-')
                aligned_seq2.append(self.seq2[j-1] if j > 0 else '-')
                j -= 1
            elif direction == 'stop' or direction == 'start':
                break
            else:
                break
            
            if i == 0 and j == 0:
                break
            if self.algorithm == "smith_waterman" and self.matrix[i][j] == 0:
                break
        
        # Reverse sequences to get correct order
        self.aligned_seq1 = ''.join(reversed(aligned_seq1))
        self.aligned_seq2 = ''.join(reversed(aligned_seq2))
        
        if self.algorithm == "needleman_wunsch":
            self.alignment_score = self.matrix[len(self.seq1)][len(self.seq2)]
        
        return self.aligned_seq1, self.aligned_seq2, trace_path[::-1]
    
    def format_alignment(self):
        """Create formatted alignment with match/mismatch markers"""
        markers = []
        for a, b in zip(self.aligned_seq1, self.aligned_seq2):
            if a == b and a != '-':
                markers.append('|')
            elif a != '-' and b != '-':
                markers.append('.')
            else:
                markers.append(' ')
        
        return self.aligned_seq1, self.aligned_seq2, ''.join(markers)
    
    def get_complexity(self):
        """Calculate and return algorithm complexity"""
        n, m = len(self.seq1), len(self.seq2)
        return {
            'time_complexity': f'O({n} × {m}) = O({n*m})',
            'space_complexity': f'O({n} × {m}) = O({n*m})' if self.algorithm == "needleman_wunsch" else f'O({n} × {m}) = O({n*m})',
            'explanation': 'Dynamic programming fills a matrix of size (m+1) × (n+1)',
            'algorithm_type': 'Global' if self.algorithm == "needleman_wunsch" else 'Local'
        }


app = Flask(__name__)
app.secret_key = 'bioinformatics_aligner_secret_key_2024'

@app.route('/')
def index():
    """Render the main input page"""
    return render_template('index.html')

@app.route('/visualize', methods=['POST'])
def visualize():
    """Process sequences and prepare visualization"""
    try:
        seq1 = request.form.get('seq1', '').strip()
        seq2 = request.form.get('seq2', '').strip()
        algorithm = request.form.get('algorithm', 'needleman_wunsch')
        match_score = int(request.form.get('match_score', 1))
        mismatch_penalty = int(request.form.get('mismatch_penalty', -1))
        gap_penalty = int(request.form.get('gap_penalty', -1))
        
        # Validate sequences
        if not seq1 or not seq2:
            return jsonify({'error': 'Please enter both sequences'}), 400
        
        # Validate sequence content (DNA/Protein)
        valid_chars = set('ACDEFGHIKLMNPQRSTVWY')
        if not (set(seq1.upper()) <= valid_chars and set(seq2.upper()) <= valid_chars):
            return jsonify({'error': 'Sequences should contain only valid amino acid or nucleotide letters'}), 400
        
        # Create aligner instance
        aligner = SequenceAligner(seq1, seq2, match_score, mismatch_penalty, gap_penalty, algorithm)
        
        # Initialize and fill matrix
        aligner.create_matrices()
        matrix, steps = aligner.fill_matrix()
        
        # Perform traceback
        aligned1, aligned2, trace_path = aligner.traceback()
        formatted_seq1, formatted_seq2, markers = aligner.format_alignment()
        
        # Get complexity info
        complexity = aligner.get_complexity()
        
        # Prepare visualization data
        viz_data = {
            'seq1': seq1,
            'seq2': seq2,
            'algorithm': algorithm,
            'match_score': match_score,
            'mismatch_penalty': mismatch_penalty,
            'gap_penalty': gap_penalty,
            'matrix': matrix,
            'steps': steps,
            'trace_path': trace_path,
            'aligned_seq1': aligned1,
            'aligned_seq2': aligned2,
            'markers': markers,
            'alignment_score': aligner.alignment_score,
            'complexity': complexity
        }
        
        # Store in session for export
        session['alignment_data'] = viz_data
        
        return render_template('visualization.html', data=viz_data)
    
    except Exception as e:
        return jsonify({'error': str(e), 'trace': tb.format_exc()}), 500

@app.route('/export')
def export():
    """Export alignment result as text file"""
    data = session.get('alignment_data', {})
    if not data:
        return "No alignment data available", 404
    
    export_text = f"""
Bioinformatics Sequence Alignment Result
=========================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Input Sequences:
Sequence 1: {data['seq1']}
Sequence 2: {data['seq2']}

Algorithm: {data['algorithm'].replace('_', ' ').title()}
Scoring Parameters:
- Match Score: {data['match_score']}
- Mismatch Penalty: {data['mismatch_penalty']}
- Gap Penalty: {data['gap_penalty']}

Alignment Result:
{data['aligned_seq1']}
{data['markers']}
{data['aligned_seq2']}

Alignment Score: {data['alignment_score']}

Time Complexity: {data['complexity']['time_complexity']}
Space Complexity: {data['complexity']['space_complexity']}
Algorithm Explanation: {data['complexity']['explanation']}
"""
    
    response = app.response_class(
        response=export_text,
        status=200,
        mimetype='text/plain'
    )
    response.headers['Content-Disposition'] = 'attachment; filename=alignment_result.txt'
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
