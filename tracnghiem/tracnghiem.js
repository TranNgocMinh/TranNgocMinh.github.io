
$(document).ready(function () {
    // Creating questionss and answers
    //*****************************************************************************
    var question1 = {
        question: "1. CSS là viết tắt của:",
        answers: ["a. Cascading Style Sheet",
"b. Cascading Style Sheets",
"c. Cascade Style Sheets",
"d. Cascading StyleSheets"],
        correct: 1
    }

    var question2 = {
        question: "2. color, background-color gọi là:",
        answers: ["a. Các thuộc tính CSS",
"b. Giá trị thuộc tính CSS",
"c. Thẻ HTML",
"d. Thuộc tính thẻ HTML"],
        correct: 0
    }
 
var question3 = {
        question: '3. Thuộc tính color dùng để định nghĩa:',
        answers: ['a. Màu nền',
'b. Màu chữ',
'c. Cả màu chữ và màu nền',
'd. Tất cả đáp án trên đều đúng'],
        correct: 1
    }
    
 var question4 = {
        question: '4. Định nghĩa CSS dùng:',
        answers: ['a. Thuộc tính style',
'b. Thẻ style',
'c. Thẻ link',
'd. a, b, c đều đúng'],
        correct: 3
    }

var question5 = {
        question: "5. Giá trị của thuộc tính text-align:",
        answers: ["a. Center",
"b. Left",
"c. Right",
"d. a, b, c đều đúng"],
        correct: 3
    }

var question6 = {
        question: "6. Căn lề đều hai bên (trái, phải) cho nội dung văn bản chúng ta dùng giá trị gì của thuộc tính text-align:",
        answers: ["a. Center",
"b. Left",
"c. Justify",
"d. Right"],
        correct: 2
    }

var question7 = {
        question: "7. Giá trị của thuộc tính border quyết định kiểu viền trong khai báo border: 1px solid blue là:",
        answers: ["a. solid",
"b. 1px",
"c. blue",
"d. Không có giá trị kiểu viền"],
        correct: 0
    }

 var question8 = {
        question: "8. Giá trị 1px của thuộc tính border trong khai báo border: 1px solid blue có ý nghĩa gì?",
        answers: ["a. Kích thước viền",
"b. Độ dày khung viền",
"c. Kiểu viền",
"d. Cả 3 đáp án trên đều đúng"],
        correct: 1
    }

var question9 = {
        question: "9. Sử dụng thuộc tính CSS nào sau đây để in đậm (bold) văn bản:",
        answers: ["a. font-size",
"b. font-style",
"c. font-family",
"d. font-weight"],
        correct: 3
    }

var question10 = {
        question: "10. Định dạng nội dung văn bản trang web sang kiểu Times New Roman dùng thuộc tính CSS:",
        answers: ["a. font-size",
"b. font-style",
"c. font-family",
"d. font-weight"],
        correct: 2
    };
    // create an array of objects

    var questions = [question1, question2, question3,question4, question5, question6, question7, question8, question9, question10];

    // Initialize variables
    //------------------------------------------------------------------

    var tags;
    var tagsClass = '';
    var liTagsid = [];
    var correctAns = 0;
    var quizPage = 1;


    var currentIndex = 0;
    var currentQuestion = questions[currentIndex];

    var prevousQuestion;
    var previousIndex = 0;

    var ulTag = document.getElementsByTagName('ul')[0];
    var button = document.getElementById('submit');
    var questionTitle = document.getElementById('question');

    //save class name so it can be reused easily
    //if I want to change it, I have to change it one place
    var classHighlight = 'selected';


    // Display Answers and hightlight selected item
    //------------------------------------------------------------------
    function showQuestions() {

        if (currentIndex != 0) {
            // create again submit button only for next pages
            ulTag.innerHTML = '';
            button.innerHTML = 'Xác nhận';
            button.className = 'submit';
            button.id = 'submit';

            //update the number of questions displayed
            document.getElementById('quizNumber').innerHTML = quizPage;
        }

        //Display Results in the final page
        if (currentIndex == (questions.length)) {
            ulTag.innerHTML = '';
            document.getElementById('question').innerHTML = '';

            showResults();

            return
        }

        questionTitle.innerHTML = currentQuestion.question;
        console.log(currentQuestion.question);

        // create a for loop to generate the answers and display them in the page
        for (var i = 0; i < currentQuestion.answers.length; i++) {
            // creating answers
            var newAns = document.createElement('li');
            newAns.id = 'ans' + (i + 1);
            newAns.className = "notSelected";
            var textAns = document.createTextNode(currentQuestion.answers[i]);
            newAns.appendChild(textAns);
            var addNewAnsHere = document.getElementById('answer');
            addNewAnsHere.appendChild(newAns);

            console.log(currentQuestion.answers[i]);
        }


        //.click() will return the result of $('.notSelected')
        var $liTags = $('.notSelected').click(function (list) {
            list.preventDefault();
            //run removeClass on every element
            //if the elements are not static, you might want to rerun $('.notSelected')
            //instead of the saved $litTags
            $liTags.removeClass(classHighlight);
            //add the class to the currently clicked element (this)
            $(this).addClass(classHighlight);

            //get id name of clicked answer
            for (var i = 0; i < currentQuestion.answers.length; i++) {
                // console.log(liTagsid[i]);
                if ($liTags[i].className == "notSelected selected") {
                    //store information to check answer
                    tags = $liTags[i].id;
                    // tagsClass = $LiTags.className;
                    console.log(tags);
                    tagsClassName = $liTags[i];
                }
            }
        });

        //check answer once it has been submitted
        button.onclick = function () { checkAnswer() };
    }

    //self calling function
    showQuestions();


    // Show Correct Answer
    //------------------------------------------------------------------
    function checkAnswer() {
        // get selected list
        var selectedItem = document.getElementById(tags);

        // check that an answer has been selected
        if (selectedItem == undefined) {
            alert("Bạn phải chọn 1 câu trả lời!")
            return
        } else {
            // get user answer if form of text
            var userAns = selectedItem.innerHTML;
        }

        // change the background of the answer according to the Results
        if (userAns == currentQuestion.answers[currentQuestion.correct]) {
            console.log("Correct! The answer is: " + userAns);
            // change color of selected item by changing className
            selectedItem.className = 'correct';
            // count the number of correct answers
            correctAns++;
            console.log(correctAns);
        } else {
            console.log("Wrong! The corrent answer is: " + currentQuestion.answers[currentQuestion.correct]);
            //change the background of the wrong answer
            selectedItem.className = 'wrong';
            //hightlight the right answer if the user got it wrong
            //change the class name of the correct answer
            ulTag.getElementsByTagName('li')[currentQuestion.correct].className = 'correct';

            console.log(currentQuestion.answers[currentQuestion.correct]);
        }

        // Create a next Question button once the answer has been submitted
        button.innerHTML = 'Câu kế tiếp';
        button.className = 'next';
        button.id = 'next';

        prevousQuestion = currentQuestion;
        quizPage++;
        currentIndex++;
        currentQuestion = questions[currentIndex];

        // Start with the next question once the "Next" button has been clicked
        button.onclick = function () { showQuestions() };
        return
    }

    // Final score
    //------------------------------------------------------------------
    function showResults() {
        //deleting page number
        document.getElementById('pages').innerHTML = '';

        // Change Title
        questionTitle.innerHTML = '<h1>Tỉ lệ chính xác</h1>';

        // Get the area that will be used to display the user's score
        var newInfo = document.getElementById('quiz-results');
        //Change the id and className of the area for the circle
        newInfo.innerHTML = '';
        newInfo.id = 'circle';
        newInfo.className = 'circle';


        //Create a Div for the fill element
        var newDiv = document.createElement('div');
        newDiv.className = 'fill';
        var addHere = document.getElementById('circle');
        addHere.appendChild(newDiv);

        // add the score to the circle
        var newScore = document.createElement('h3');
        newScore.className = 'score';
        var textScore = document.createTextNode(Math.floor((correctAns / questions.length) * 100) + '%');
        newScore.appendChild(textScore);
        addHere.appendChild(newScore);

        //use jquary to grab the text of the score
        var score = $(".score").text();

        //fill the circle in base of the score
        $(".fill").css("height", score);

        if (correctAns == 10) {
            var newCongrats = document.createElement('p');
            newCongrats.id = "red";
            var textCongrats = document.createTextNode('Chúc mừng bạn đã hoàn thành. Bạn làm tốt lắm! ');
            newCongrats.appendChild(textCongrats);
            var a = document.createElement('a');
            var linkText = document.createTextNode("Trở lại bài học >");
            a.appendChild(linkText);
            a.title = "Trở lại bài học";
            a.href = "https://hoctructuyencntt.com/2021/07/04/nhap-mon-css/";
            document.getElementById('display-area-01').appendChild(newCongrats);
            document.getElementById('display-area-01').appendChild(a);
            //confettiEffect();
        }
        else {
            var newCongrats = document.createElement('p');
            newCongrats.id = "red";
            var textCongrats = document.createTextNode('Bạn chưa hoàn thành các câu hỏi. Nhấn F5 thực hiện lại.')
            newCongrats.appendChild(textCongrats);
            document.getElementById('display-area-01').appendChild(newCongrats);

            //confettiEffect();
        }

    }

    // Confetti Effect by Gtibo "Confetti Party"
    //------------------------------------------------------------------
    function confettiEffect() {
        //grabing area to create the effect
        canvas = document.getElementById("canvas");
        context = canvas.getContext("2d");
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;

        // creating the tabel
        particle = [];
        particleCount = 0,
            gravity = 0.3,
            colors = [
                '#f44336', '#e91e63', '#9c27b0', '#673ab7', '#3f51b5',
                '#2196f3', '#03a9f4', '#00bcd4', '#009688', '#4CAF50',
                '#8BC34A', '#CDDC39', '#FFEB3B', '#FFC107', '#FF9800',
                '#FF5722', '#795548'
            ];

        for (var i = 0; i < 300; i++) {

            particle.push({
                x: width / 2,
                y: height / 2,
                boxW: randomRange(5, 20),
                boxH: randomRange(5, 20),
                size: randomRange(2, 8),

                spikeran: randomRange(3, 5),

                velX: randomRange(-8, 8),
                velY: randomRange(-50, -10),

                angle: convertToRadians(randomRange(0, 360)),
                color: colors[Math.floor(Math.random() * colors.length)],
                anglespin: randomRange(-0.2, 0.2),

                draw: function () {
                    context.save();
                    context.translate(this.x, this.y);
                    context.rotate(this.angle);
                    context.fillStyle = this.color;
                    context.beginPath();

                    context.fillRect(this.boxW / 2 * -1, this.boxH / 2 * -1, this.boxW, this.boxH);
                    context.fill();
                    context.closePath();
                    context.restore();
                    this.angle += this.anglespin;
                    this.velY *= 0.999;
                    this.velY += 0.3;

                    this.x += this.velX;
                    this.y += this.velY;

                    if (this.y < 0) {
                        this.velY *= -0.2;
                        this.velX *= 0.9;
                    };

                    if (this.y > height) {
                        this.anglespin = 0;
                        this.y = height;
                        this.velY *= -0.2;
                        this.velX *= 0.9;
                    };

                    if (this.x > width || this.x < 0) {
                        this.velX *= -0.5;
                    };
                },
            });
        }

        function drawScreen() {
            context.globalAlpha = 1;
            for (var i = 0; i < particle.length; i++) {
                particle[i].draw();
            }
        }

        function loadImage(url) {
            var img = document.createElement("img");
            img.src = url;
            return img;
        }

        function update() {
            context.clearRect(0, 0, width, height);
            drawScreen();
            requestAnimationFrame(update);
        }

        update();

        function randomRange(min, max) {
            return min + Math.random() * (max - min);
        }

        function randomInt(min, max) {
            return Math.floor(min + Math.random() * (max - min + 1));
        }

        function convertToRadians(degree) {
            return degree * (Math.PI / 180);
        }

        function drawStar(cx, cy, spikes, outerRadius, innerRadius, color) {
            var rot = Math.PI / 2 * 3;
            var x = cx;
            var y = cy;
            var step = Math.PI / spikes;

            context.strokeSyle = "#000";
            context.beginPath();
            context.moveTo(cx, cy - outerRadius)
            for (i = 0; i < spikes; i++) {
                x = cx + Math.cos(rot) * outerRadius;
                y = cy + Math.sin(rot) * outerRadius;
                context.lineTo(x, y)
                rot += step

                x = cx + Math.cos(rot) * innerRadius;
                y = cy + Math.sin(rot) * innerRadius;
                context.lineTo(x, y)
                rot += step
            }

            context.lineTo(cx, cy - outerRadius)
            context.closePath();
            context.fillStyle = color;
            context.fill();

        }
    }
    $("#img").hide();
    $("textarea").hide();
    $("#html").click(function () {
        $(this).hide();
        $("#img").show();
        $("img").hide();
        $("textarea").show();
    });
    $("#img").click(function () {
        $(this).hide();
        $("#html").show();
        $("img").show();
        $("textarea").hide();
    });
});