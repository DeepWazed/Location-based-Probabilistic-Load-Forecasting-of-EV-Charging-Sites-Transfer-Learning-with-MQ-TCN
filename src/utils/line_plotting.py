import matplotlib.pyplot as plt
import plotly.express as px

def create_line_plot(result_df, title):
    final_result = result_df.copy()
    # set connectionTime as index
    final_result.set_index('connectionTime', inplace=True)

    # create line plot
    fig = px.line(final_result, x=final_result.index, y=['y_test_org', 'y_test_pred'],
                  labels={'value': 'kWh Delivered', 'variable': 'Data'},
                  title=title)

    # customize layout
    fig.update_layout(xaxis_title='Date', yaxis_tickformat='.2f')

    # show plot
    fig.show()