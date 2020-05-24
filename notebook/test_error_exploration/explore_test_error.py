def explore_test_error(X_test, y_test,  final_predictions ):
    X_test['pred'] = final_predictions.tolist()

    X_test['real_value'] = y_test.values
    X_test['diff_error'] = X_test.apply(
        lambda row: abs(row.pred - row.real_value), axis=1)

    #save error per district to csv:
    # X_test.to_csv('out.csv', index=False)


    #describe the diffferent total X_TEST features statistics:
    total_test_stats = X_test.describe()

    #describe the top 50 worst error districts features:
    worst250_df = X_test.nlargest(250, 'diff')
    dd250_stats = worst250_df.describe()