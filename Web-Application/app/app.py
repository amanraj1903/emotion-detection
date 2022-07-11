from flask import Flask, render_template, request
import testpy

app = Flask(__name__)

#index page
@app.route("/")
@app.route("/main.html")
def index():
    return render_template('main.html')

#results page
@app.route('/result.html',methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
        # print(request.form['result'])
        prediction = testpy.prediction_by_list(request.form['result'])
        prediction_all = testpy.prediction_overall(request.form['result'])
        return render_template('result.html', value=prediction,overall=prediction_all)

#main
if __name__ == "_main_":
    app.run(debug=True)
