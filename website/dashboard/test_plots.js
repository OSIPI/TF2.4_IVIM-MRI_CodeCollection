document.addEventListener('DOMContentLoaded', function() {
    fetch('report-summary.json')
        .then(response => response.json())
        .then(data => {
            function createPlot(container, parameter) {
                var reference_values = data.map(d => d[parameter]);
                var fit_values = data.map(d => d[parameter + '_fit']);
                var errors = fit_values.map((d, i) => d - reference_values[i]);
                var minRefValue = Math.min(...reference_values);
                var maxRefValue = Math.max(...reference_values);
                // Create a range for tolerance trace x axis.
                var xRange = [minRefValue, maxRefValue];

                var tolerance = xRange.map((d, i) => data[reference_values.indexOf(d)]['atol'][parameter] + data[i]['rtol'][parameter] * d);
                var negative_tolerance = tolerance.map(t => -t);

                var scatter_trace = {
                    x: reference_values,
                    y: errors,
                    mode: 'markers',
                    type: 'scatter',
                    name: parameter.toUpperCase() + ' Fit Errors'
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
