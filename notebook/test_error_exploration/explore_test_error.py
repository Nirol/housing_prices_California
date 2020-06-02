def explore_test_error(X_test_prepared, y_test,  final_predictions ):
    X_test_prepared['pred'] = final_predictions.tolist()

    X_test_prepared['real_value'] = y_test.values
    X_test_prepared['diff_error'] = X_test_prepared.apply(
        lambda row: abs(row.pred - row.real_value), axis=1)

    #save error per district to csv:
    # X_test.to_csv('out.csv', index=False)


    #describe the diffferent total X_TEST features statistics:
    total_test_stats = X_test_prepared.describe()
    total_test_stats.T.to_csv('total_test_set_stat_v3.csv', index=True)
    #describe the top 50 worst error districts features:
    worst250_df = X_test_prepared.nlargest(250, 'diff_error')
    dd250_stats = worst250_df.describe()
    dd250_stats.T.to_csv('worst250_test_set_stat_v3.csv', index=True)

