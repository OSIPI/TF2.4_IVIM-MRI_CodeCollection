document.addEventListener('DOMContentLoaded', function() {
    fetch('report-summary.json')
        .then(response => response.json())
        .then(data => {
            function createPlot(container, parameter) {
                var reference_values = data.map(d => d[parameter]);
                var fit_values = data.map(d => d[parameter + '_fit']);
                var errors = fit_values.map((d, i) => d - reference_values[i]);

                var tolerance = reference_values.map((d, i) => data[i]['atol'][parameter] + data[i]['rtol'][parameter] * d);
                var negative_tolerance = tolerance.map(t => -t);

                var minRefValue = Math.min(...reference_values);
                var maxRefValue = Math.max(...reference_values);
                // Create a range for tolerance trace x axis.
                var xRange = [minRefValue, maxRefValue];
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
                    mode: 'lines',
                    type: 'scatter',
                    line: { dash: 'dash', color: 'red' },
                    name: 'Positive Tolerance'
                };

                var negative_tolerance_trace = {
                    x: xRange,
                    y: negative_tolerance,
                    mode: 'lines',
                    type: 'scatter',
                    line: { dash: 'dash', color: 'green' },
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
