// $(document).ready(function(){
//
//     var csrf = $("input[name=csrfmiddlewaretoken]").val();
//
//     $('#uploadVideoBtn').on('click','button',function(){
//         $.ajax({
//             url: '/video/',
//             type: 'post',
//             data: {
//                 keyWhatEverYouWant: $(this).text(),
//                 csrfmiddlewaretoken: csrf
//             },
//             success: function (response) {
//                 $('#uploadVideoBtn').text(response.videos)
//             },
//             failure: function(response) {
//                 alert('Got an error dude');
//             }
//         });
//     });
// }

// //For checking whether jquery is running or not.
// window.onload = function() {
//     if (window.jQuery) {
//         // jQuery is loaded
//         alert("Yeah!");
//
//     } else {
//         // jQuery is not loaded
//         alert("Doesn't Work");
//     }
// }

$(document).ready(function () {

    $('#imageLoadingView').hide();
    $('#imageLoadingText').hide();
    $('#loadingView').hide();
    $('#LoadingText').hide();
    $('#videoPlayerView').hide();
    $('#videoCloseBtn').hide();
    $('#videoPlayerContainer').hide();
    $('#loadingViewLiveFeed').hide();
    $('#loadingViewLiveFeedText').hide();
    $('#stopLiveFeedBtn').hide();

    $('button#uploadVideoBtn').on('click', function () {
        var myForm = $("form#videoForm");
        if (myForm) {
            if($('#videoUrlId')[0].files.length === 0){
                alert("Please select a video.")
            }else {
                $(this).prop('disabled', true);
                $(myForm).submit();
                $("#loadingView").show();
                $("#LoadingText").show();
            }
        }
    });

    $('#List>li').on('click',function() {
        $(this).addClass('active');
    });
});

function playVideo(videoTitle,id){
    $('#videoForPlayBtn-'+id).on('click',function(){
        $('#videoPlayerContainer').show();
        $('#videoPlayerView').show();
        $('#videoCloseBtn').show();
        $('#videoPlayerView source').attr('src','/media/videos/detections/'+videoTitle);
        $("#videoPlayerView")[0].load();
    });
}

function closeVideoPlayer(){
    $('#videoCloseBtn').on('click',function(){
        $("#videoPlayerView")[0].pause();
        $("#videoPlayerView").hide();
        $('#videoPlayerContainer').hide();
        $(this).hide();
    });
}

function deleteImageByUser(id){
    var action = confirm("Do you really wants to delete this image?");
    var endpoint = $("#deleteImageByUserBtn-"+id).attr("data-url");
    var csrf_token = $('input[name=csrfmiddlewaretoken]').val();
    var data = {}
    data['csrfmiddlewaretoken'] = csrf_token;
    if(action != false){
        $.ajax({
            type: "POST",
            url: endpoint,
            data: data,
            dataType: 'json',
            success: function(data){
                if(data.deleted){
                    $("#imageUserRow-"+id).remove();
                }
            },
        });
    }
}

function deleteVideoByUser(id){
    var action = confirm("Do you really want to delete this video?");
    if(action != false){
        var endpoint = $("#deleteMyVideoByUserBtn-"+id).attr("data-url");
        var csrf_token = $('input[name=csrfmiddlewaretoken]').val();
        var data = {}
        data['csrfmiddlewaretoken'] = csrf_token;

        $.ajax({
            type: "POST",
            url: endpoint,
            data: data,
            dataType: 'Json',
            success: function(data){
                if(data.deleted){
                    $('#deleteUserVideoRow-'+id).remove();
                }
            }
        });
    }
}

function deleteImageByAdmin(id){
    var action = confirm("Do you really wants to delete this image?");
    var endpoint = $("#deleteImageBtn-"+id).attr("data-url");
    var csrf_token = $('input[name=csrfmiddlewaretoken]').val();
    var data = {}

    data['csrfmiddlewaretoken'] = csrf_token;

    if(action != false){
        $.ajax({
            type: "POST",
            url: endpoint,
            data: data,
            dataType: 'json',
            success: function (data) {
                if (data.deleted) {
                  $("#imageRow-" + id).remove();
                }
            },
        });
    }
}

function deleteVideoByAdmin(id){
    var action = confirm("Do you really wants to delete this video?");
    var endpoint = $("#deleteVideoByAdminBtn-"+id).attr("data-url");
    var csrf_token = $('input[name=csrfmiddlewaretoken]').val();
    var data = {}
    data['csrfmiddlewaretoken'] = csrf_token;

    if(action != false){
        $.ajax({
            type: "POST",
            url: endpoint,
            data: data,
            dataType: 'json',
            success: function(data){
                if(data.deleted){
                    $("#userUploadedVideoRow-"+id).remove();
                }
            },
        });
    }
}

function deleteUserByAdmin(id){
    var currentCount = 0;
    var action = confirm("Do you really want to delete this user? ");
    var endpoint = $("#deleteUserBtn-"+id).attr("data-url");
    var csrf_token = $("input[name=csrfmiddlewaretoken]").val()
    var data = {}
    data['csrfmiddlewaretoken'] = csrf_token
    if(action != false){
        $.ajax({
            type: "POST",
            url: endpoint,
            data: data,
            dataType: 'Json',
            success: function(data){
                if(data.deleted){
                    $("#deleteTableRow-"+id).remove();
                    currentCount += parseInt(document.getElementById('totalUsersCount').innerHTML);
                    currentCount -= 1;
                    document.getElementById('totalUsersCount').innerHTML = currentCount;
                }else{

                }
            },
        });
    }
}


function addNewUser(form){
    var action = confirm("Do you really want to add this user? ");
    var currentCount = 0;
    if(action != false){
        $.ajax({
            url: 'add/user/',
            type: "POST",
            data: $(form).serialize(),
            dataType: 'Json',
            success: function(data){
                if(data.created){
                    alert(data.message);
                    $('#addUserModel').modal('toggle');
                    currentCount += parseInt(document.getElementById('totalUsersCount').innerHTML);
                    currentCount += 1;
                    document.getElementById('totalUsersCount').innerHTML = currentCount;
                }else{
                    alert(data.message);
                    $('#addUserModel').modal('toggle');
                }
            },
        });
    }
}

function liveFeedVideoURL(){
    var feed_url = $("input[name=feedURL]").val();
    var csrf_token = $("input[name=csrfmiddlewaretoken]").val();
    var data = {}
    data['csrfmiddlewaretoken'] = csrf_token
    data['feedURL'] = feed_url
    $.ajax({
        url: '/video/feed/',
        type: 'POST',
        data: data,
        success: function(data){
            console.log(data);
            path = data.path
            path = path.replaceAll('/',"_")
            $('#live_feed_player').show();
            $('#live_feed_player').attr('src','/video/trial/feed/'+data.scheme+"_"+data.netloc+path+'/');
            $('#loadingViewLiveFeed').show();
            $('#loadingViewLiveFeedText').show();
            $('#startLiveFeedBtn').hide();
            $('#stopLiveFeedBtn').show();
        },
    });
}


function stopLiveFeed(){
    var csrf_token = $("input[name=csrfmiddlewaretoken]").val();
    var data = {}
    data['csrfmiddlewaretoken'] = csrf_token
    data['stopFeedBool'] = true
    $.ajax({
       url: '/video/feed/',
       type: 'POST',
       data: data,
       success: function(data){
              $('#live_feed_player').attr('src','/video/trial/feed/'+data.stopFeedBool+'/');
              $('#startLiveFeedBtn').show();
              $('#stopLiveFeedBtn').hide();
              $('#loadingViewLiveFeed').hide();
              $('#loadingViewLiveFeedText').hide();
              $('#live_feed_player').hide();
       },
    });
}

function showProgress() {
		$('#imageLoadingView').show();
		$('#imageLoadingText').show();
}

/**
######################## please dont delete this comment this is very important comment ############################

//formData += "&parentUser="+encodeURIComponent($(currentUser).serialize());

$('#live_feed_player2').attr('src','data:image/jpeg;base64,'+data.images.replace('\"',''));

####################################################################################################################
**/