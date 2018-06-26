################################################################################
#
#
# Interactive tools, specially for exploring data and results of models
#
#
################################################################################


from mpld3 import plugins


suggested_css = """
table
{
  border-collapse: collapse;
}
th
{
  color: #ffffff;
  background-color: #000000;
}
td
{
  background-color: #cccccc;
}
table, th, td
{
  font-family:Arial, Helvetica, sans-serif;
  border: 1px solid black;
  text-align: right;
}
"""


def create_tooltip(plot_obj, df, col_names, css=None):
    """

    :param plot_obj: matplotlib plot
    :param df: pandas dataframe with the data
    :param col_names: column names to be displayed in the tooltip
    :param css: css for the displayed table
    :return: PointHtmlTooltip
    """
    labels = []
    for i in range(len(df)):
        label = df.iloc[i][col_names].T.to_frame()
        label.columns = ['Obs {0}'.format(i)]
        labels.append(str(label.to_html()))
    return plugins.PointHTMLTooltip(plot_obj, labels,
                                    voffset=10, hoffset=10, css=css)

