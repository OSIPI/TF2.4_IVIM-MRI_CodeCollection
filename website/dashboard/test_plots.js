document.addEventListener('DOMContentLoaded', function() {
    fetch('test_results_report.json')
        .then(response => response.json())
        .then(data => {
            var rtol = data['rtol']

            var atol = data['atol']

            function createPlot(container, parameter, rtol_value, atol_value) {
                var reference_values = data['results'].map(d => d[parameter]); // Assuming fit values are the reference
                var fit_values = data['results'].map(d => d[parameter + '_fit']);
                var errors = fit_values.map((d, i) => d - reference_values[i]);

                var tolerance = reference_values.map(d => atol_value + rtol_value * d);
                var negative_tolerance = reference_values.map(d => -(atol_value + rtol_value * d));

                var scatter_trace = {
                    x: reference_values,
                    y: errors,
                    mode: 'markers',
                    type: 'scatter',
                    name: parameter.toUpperCase() + ' Fit Errors'
                };

                var tolerance_trace = {
                    x: reference_values,
                    y: tolerance,
                    mode: 'lines',
                    type: 'scatter',
                    line: { dash: 'dash', color: 'red' },
                    name: 'Positive Tolerance'
                };

                var negative_tolerance_trace = {
                    x: reference_values,
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

            createPlot('plot_f_fit', 'f', rtol.f, atol.f);
            createPlot('plot_Dp_fit', 'Dp', rtol.Dp, atol.Dp);
            createPlot('plot_D_fit', 'D', rtol.D, atol.D);
        });
});
