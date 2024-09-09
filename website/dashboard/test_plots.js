document.addEventListener('DOMContentLoaded', function() {
    fetch('report-summary.json')
        .then(response => response.json())
        .then(data => {
            function createPlot(container, parameter) {
                var reference_values = data.map(d => d[parameter]);
                var fit_values = data.map(d => d[parameter + '_fit']);
                var minRefValue = Math.min(...reference_values);
                var maxRefValue = Math.max(...reference_values);
                // Create a range for tolerance trace x axis.
                var xRange = [minRefValue, maxRefValue];
                var DefaultTolerance = {
                    "rtol": {
                        "f": 0.05,
                        "D": 2,
                        "Dp": 0.5
                    },
                    "atol": {
                        "f": 0.2,
                        "D": 0.001,
                        "Dp": 0.06
                    }
                }
                var tolerance = xRange.map((d) => DefaultTolerance['atol'][parameter] + DefaultTolerance['rtol'][parameter] * d);
                var negative_tolerance = tolerance.map(t => -t);

                var errors = fit_values.map((d, i) => (d - reference_values[i]));

                // Define colors for each status
                var statusColors = {
                    'passed': 'green',
                    'xfailed': 'yellow',
                    'failed': 'red'
                };

                // Assign color based on the status
                var marker_colors = data.map(entry => statusColors[entry.status]);
            
                var scatter_trace = {
                    x: reference_values,
                    y: errors,
                    mode: 'markers',
                    type: 'scatter',
                    name:  `${parameter} fitting values`,
                    text: data.map(entry => `Algorithm: ${entry.algorithm} Region: ${entry.name}`),
                    marker: {
                        color: marker_colors
                    }
                };

                var tolerance_trace = {
                    x: xRange,
                    y: tolerance,
                    type: 'line',
                    line: { dash: 'dash', color: 'green' },
                    name: 'Positive Tolerance'
                };

                var negative_tolerance_trace = {
                    x: xRange,
                    y: negative_tolerance,
                    type: 'line',
                    line: { dash: 'dash', color: 'red' },
                    name: 'Negative Tolerance'
                };

                var layout = {
                    title: `Error Plot for ${parameter.toUpperCase()}_fit with Tolerance Bands`,
                    xaxis: { title: `Reference ${parameter.toUpperCase()} Values` },
                    yaxis: { title: `Error (${parameter}_fit - Reference ${parameter})` }
                };

                var plot_data = [scatter_trace, tolerance_trace, negative_tolerance_trace];

                Plotly.newPlot(container, plot_data, layout);
            }

            createPlot('plot_f_fit', 'f');
            createPlot('plot_Dp_fit', 'Dp');
            createPlot('plot_D_fit', 'D');
        });
});
