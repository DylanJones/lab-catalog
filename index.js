
var express = require('express')
var path = require('path');
var app = express();

app.set('port', process.env.PORT || 8080 );
app.set('view engine', 'hbs');

app.use('/js', express.static(path.join(__dirname, '/js')));
app.use('/css', express.static(path.join(__dirname, '/css')));

app.get('/', function(req, res){
    res.render("main");
});

app.get('/not_a_search', function(req, res){
    var theQuery = req.query.q;
    res.send('query parameter:' + theQuery);
});



var listener = app.listen(app.get('port'), function() {
  console.log( 'Express server started on port: '+listener.address().port );
});