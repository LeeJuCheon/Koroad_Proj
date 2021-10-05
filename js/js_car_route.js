var map;
var markerInfo;
//출발지,도착지 마커
var marker_s, marker_e, marker_p;
var marker1, marker2;

//경로그림정보
var drawInfoArr = [];
var drawInfoArr2 = [];

var chktraffic = [];
var resultdrawArr = [];
var resultMarkerArr = [];
var loArray = [];
var laArray = [];

function initTmap() {
  // 1. 지도 띄우기
  map = new Tmapv2.Map("map_div", {
    center: new Tmapv2.LatLng(37.49241689559544, 127.03171389453507),  //시작 지도 좌표
    width: "100%",
    height: "400px",
    zoom: 11,   //지도 크기 설정
    zoomControl: true,
    scrollwheel: true
  });

  // 2. 시작, 도착 심볼찍기
  // 시작
  marker_s = new Tmapv2.Marker(
    {
      position: new Tmapv2.LatLng(37.566567545861645,
        126.9850380932383),
      icon: "http://tmapapi.sktelecom.com/upload/tmap/marker/pin_r_m_s.png",
      iconSize: new Tmapv2.Size(24, 38),
      map: map
    });

  //도착
  marker_e = new Tmapv2.Marker(
    {
      position: new Tmapv2.LatLng(37.403049076341794,
        127.10331814639885),
      icon: "http://tmapapi.sktelecom.com/upload/tmap/marker/pin_r_m_e.png",
      iconSize: new Tmapv2.Size(24, 38),
      map: map
    });

  // 3. 경로탐색 API 사용요청
  $("#btn_select")
    .click(
      function () {

        //기존 맵에 있던 정보들 초기화
        resettingMap();

        //html에서 입력받는 값
        var searchOption = $("#selectLevel").val();
        var trafficInfochk = $("#year").val();
        var fullAddr1 = $("#fullAddr1").val();
        var fullAddr2 = $("#fullAddr2").val();

        //JSON fullAddr1
        $.ajax({
          method: "GET",
          url: "https://apis.openapi.sk.com/tmap/geo/fullAddrGeo?version=1&format=json&callback=result",
          async: false,
          data: {
            "appKey": "l7xxb7f655f2345248f7b5d6f198531b219a",
            "coordType": "WGS84GEO",
            "fullAddr": fullAddr1
          },
          success: function (response) {
            var resultInfo = response.coordinateInfo; // .coordinate[0];

            // 검색 결과 정보가 없을 때 처리
            if (resultInfo.coordinate.length == 0) {
              $("#result2").text(
                "요청 데이터가 올바르지 않습니다.");
            }

            else {
              var lon, lat;
              var resultCoordinate = resultInfo.coordinate[0];
              if (resultCoordinate.lon.length > 0) {
                // 구주소
                lon = resultCoordinate.lon;
                lat = resultCoordinate.lat;
              } else {
                // 신주소
                lon = resultCoordinate.newLon;
                lat = resultCoordinate.newLat
              }

              var lonEntr, latEntr;

              if (resultCoordinate.lonEntr == undefined && resultCoordinate.newLonEntr == undefined) {
                lonEntr = 0;
                latEntr = 0;
              } else {
                if (resultCoordinate.lonEntr.length > 0) {
                  lonEntr = resultCoordinate.lonEntr;
                  latEntr = resultCoordinate.latEntr;
                } else {
                  lonEntr = resultCoordinate.newLonEntr;
                  latEntr = resultCoordinate.newLatEntr;
                }
              }
              loArray.push(lon);
              laArray.push(lat);
              var markerPosition1 = new Tmapv2.LatLng(Number(lat), Number(lon));

              // 검색 결과 표출
              var matchFlag, newMatchFlag;

              // 검색 결과 주소를 담을 변수
              var address = '', newAddress = '';
              var city, gu_gun, eup_myun, legalDong, adminDong, ri, bunji;
              var buildingName, buildingDong, newRoadName, newBuildingIndex, newBuildingName, newBuildingDong;

              //// 새주소인 경우 matchFlag가 아닌 newMatchFlag가 응답값으로 온다
              if (resultCoordinate.newMatchFlag.length > 0) {
                // 새(도로명) 주소 좌표 매칭
                // 구분 코드
                newMatchFlag = resultCoordinate.newMatchFlag;

                // 시/도 명칭
                if (resultCoordinate.city_do.length > 0) {
                  city = resultCoordinate.city_do;
                  newAddress += city + "\n";
                }

                // 군/구 명칭
                if (resultCoordinate.gu_gun.length > 0) {
                  gu_gun = resultCoordinate.gu_gun;
                  newAddress += gu_gun + "\n";
                }

                // 읍면동 명칭
                if (resultCoordinate.eup_myun.length > 0) {
                  eup_myun = resultCoordinate.eup_myun;
                  newAddress += eup_myun + "\n";
                } else {
                  // 출력 좌표에 해당하는 법정동 명칭
                  if (resultCoordinate.legalDong.length > 0) {
                    legalDong = resultCoordinate.legalDong;
                    newAddress += legalDong + "\n";
                  }
                  // 출력 좌표에 해당하는 행정동 명칭
                  if (resultCoordinate.adminDong.length > 0) {
                    adminDong = resultCoordinate.adminDong;
                    newAddress += adminDong + "\n";
                  }
                }
                // 출력 좌표에 해당하는 리 명칭
                if (resultCoordinate.ri.length > 0) {
                  ri = resultCoordinate.ri;
                  newAddress += ri + "\n";
                }
                // 출력 좌표에 해당하는 지번 명칭
                if (resultCoordinate.bunji.length > 0) {
                  bunji = resultCoordinate.bunji;
                  newAddress += bunji + "\n";
                }
                // 새(도로명)주소 매칭을 한 경우, 길 이름을 반환
                if (resultCoordinate.newRoadName.length > 0) {
                  newRoadName = resultCoordinate.newRoadName;
                  newAddress += newRoadName + "\n";
                }
                // 새(도로명)주소 매칭을 한 경우, 건물 번호를 반환
                if (resultCoordinate.newBuildingIndex.length > 0) {
                  newBuildingIndex = resultCoordinate.newBuildingIndex;
                  newAddress += newBuildingIndex + "\n";
                }
                // 새(도로명)주소 매칭을 한 경우, 건물 이름를 반환
                if (resultCoordinate.newBuildingName.length > 0) {
                  newBuildingName = resultCoordinate.newBuildingName;
                  newAddress += newBuildingName + "\n";
                }
                // 새주소 건물을 매칭한 경우 새주소 건물 동을 반환
                if (resultCoordinate.newBuildingDong.length > 0) {
                  newBuildingDong = resultCoordinate.newBuildingDong;
                  newAddress += newBuildingDong + "\n";
                }
                // 검색 결과 표출
                if (lonEntr > 0) {
                  var docs = "<a style='color:orange' href='#webservice/docs/fullTextGeocoding'>Docs</a>"
                  var text = "검색결과(새주소) : " + newAddress + ",\n 응답코드:" + newMatchFlag + "(상세 코드 내역은 " + docs + " 에서 확인)" + "</br> 위경도좌표(중심점) : " + lat + ", " + lon + "</br>위경도좌표(입구점) : " + latEntr + ", " + lonEntr;
                  $("#result2").html(text);
                } else {
                  var docs = "<a style='color:orange' href='#webservice/docs/fullTextGeocoding'>Docs</a>"
                  var text = "검색결과(새주소) : " + newAddress + ",\n 응답코드:" + newMatchFlag + "(상세 코드 내역은 " + docs + " 에서 확인)" + "</br> 위경도좌표(입구점) : 위경도좌표(입구점)이 없습니다.";
                  $("#result2").html(text);
                }
              }

              // 구주소인 경우 newMatchFlag가 아닌 MatchFlag가 응닶값으로 온다
              if (resultCoordinate.matchFlag.length > 0) {
                // 매칭 구분 코드
                matchFlag = resultCoordinate.matchFlag;

                // 시/도 명칭
                if (resultCoordinate.city_do.length > 0) {
                  city = resultCoordinate.city_do;
                  address += city + "\n";
                }
                // 군/구 명칭
                if (resultCoordinate.gu_gun.length > 0) {
                  gu_gun = resultCoordinate.gu_gun;
                  address += gu_gun + "\n";
                }
                // 읍면동 명칭
                if (resultCoordinate.eup_myun.length > 0) {
                  eup_myun = resultCoordinate.eup_myun;
                  address += eup_myun + "\n";
                }
                // 출력 좌표에 해당하는 법정동 명칭
                if (resultCoordinate.legalDong.length > 0) {
                  legalDong = resultCoordinate.legalDong;
                  address += legalDong + "\n";
                }
                // 출력 좌표에 해당하는 행정동 명칭
                if (resultCoordinate.adminDong.length > 0) {
                  adminDong = resultCoordinate.adminDong;
                  address += adminDong + "\n";
                }
                // 출력 좌표에 해당하는 리 명칭
                if (resultCoordinate.ri.length > 0) {
                  ri = resultCoordinate.ri;
                  address += ri + "\n";
                }
                // 출력 좌표에 해당하는 지번 명칭
                if (resultCoordinate.bunji.length > 0) {
                  bunji = resultCoordinate.bunji;
                  address += bunji + "\n";
                }
                // 출력 좌표에 해당하는 건물 이름 명칭
                if (resultCoordinate.buildingName.length > 0) {
                  buildingName = resultCoordinate.buildingName;
                  address += buildingName + "\n";
                }
                // 출력 좌표에 해당하는 건물 동을 명칭
                if (resultCoordinate.buildingDong.length > 0) {
                  buildingDong = resultCoordinate.buildingDong;
                  address += buildingDong + "\n";
                }
                // 검색 결과 표출
                if (lonEntr > 0) {
                  var docs = "<a style='color:orange' href='#webservice/docs/fullTextGeocoding'>Docs</a>";
                  var text = "검색결과(지번주소) : " + address + "," + "\n" + "응답코드:" + matchFlag + "(상세 코드 내역은 " + docs + " 에서 확인)" + "</br>" + "위경도좌표(중심점) : " + lat + ", " + lon + "</br>" + "위경도좌표(입구점) : " + latEntr + ", " + lonEntr;
                  $("#result2").html(text);
                } else {
                  var docs = "<a style='color:orange' href='#webservice/docs/fullTextGeocoding'>Docs</a>";
                  var text = "검색결과(지번주소) : " + address + "," + "\n" + "응답코드:" + matchFlag + "(상세 코드 내역은 " + docs + " 에서 확인)" + "</br>" + "위경도좌표(입구점) : 위경도좌표(입구점)이 없습니다.";
                  $("#result2").html(text);
                }
              }
            }

          }
          , error: function (request, status, error) {
            console.log(request);
            console.log("code:" + request.status + "\n message:" + request.responseText + "\n error:" + error);
            $("#result2").html("error");
          }
        });

        //JSON fullAddr2
        $.ajax({
          method: "GET",
          url: "https://apis.openapi.sk.com/tmap/geo/fullAddrGeo?version=1&format=json&callback=result",
          async: false,
          data: {
            "appKey": "l7xxb7f655f2345248f7b5d6f198531b219a",
            "coordType": "WGS84GEO",
            "fullAddr": fullAddr2
          },
          success: function (response) {
            var resultInfo = response.coordinateInfo; // .coordinate[0];

            // 검색 결과 정보가 없을 때 처리
            if (resultInfo.coordinate.length == 0) {
              $("#result3").text(
                "요청 데이터가 올바르지 않습니다.");
            }
            else {
              var lon, lat;
              var resultCoordinate = resultInfo.coordinate[0];
              if (resultCoordinate.lon.length > 0) {
                // 구주소
                lon = resultCoordinate.lon;
                lat = resultCoordinate.lat;
              } else {
                // 신주소
                lon = resultCoordinate.newLon;
                lat = resultCoordinate.newLat
              }

              var lonEntr, latEntr;

              if (resultCoordinate.lonEntr == undefined && resultCoordinate.newLonEntr == undefined) {
                lonEntr = 0;
                latEntr = 0;
              } else {
                if (resultCoordinate.lonEntr.length > 0) {
                  lonEntr = resultCoordinate.lonEntr;
                  latEntr = resultCoordinate.latEntr;
                } else {
                  lonEntr = resultCoordinate.newLonEntr;
                  latEntr = resultCoordinate.newLatEntr;
                }
              }
              loArray.push(lon);
              laArray.push(lat);

              var markerPosition2 = new Tmapv2.LatLng(Number(lat), Number(lon));

              // 검색 결과 표출
              var matchFlag, newMatchFlag;
              // 검색 결과 주소를 담을 변수
              var address = '', newAddress = '';
              var city, gu_gun, eup_myun, legalDong, adminDong, ri, bunji;
              var buildingName, buildingDong, newRoadName, newBuildingIndex, newBuildingName, newBuildingDong;

              // 새주소인 경우 matchFlag가 아닌 newMatchFlag가 응답값으로 온다
              if (resultCoordinate.newMatchFlag.length > 0) {
                // 새(도로명) 주소 좌표 매칭 구분 코드
                newMatchFlag = resultCoordinate.newMatchFlag;

                // 시/도 명칭
                if (resultCoordinate.city_do.length > 0) {
                  city = resultCoordinate.city_do;
                  newAddress += city + "\n";
                }

                // 군/구 명칭
                if (resultCoordinate.gu_gun.length > 0) {
                  gu_gun = resultCoordinate.gu_gun;
                  newAddress += gu_gun + "\n";
                }

                // 읍면동 명칭
                if (resultCoordinate.eup_myun.length > 0) {
                  eup_myun = resultCoordinate.eup_myun;
                  newAddress += eup_myun + "\n";
                } else {
                  // 출력 좌표에 해당하는 법정동 명칭
                  if (resultCoordinate.legalDong.length > 0) {
                    legalDong = resultCoordinate.legalDong;
                    newAddress += legalDong + "\n";
                  }
                  // 출력 좌표에 해당하는 행정동 명칭
                  if (resultCoordinate.adminDong.length > 0) {
                    adminDong = resultCoordinate.adminDong;
                    newAddress += adminDong + "\n";
                  }
                }
                // 출력 좌표에 해당하는 리 명칭
                if (resultCoordinate.ri.length > 0) {
                  ri = resultCoordinate.ri;
                  newAddress += ri + "\n";
                }
                // 출력 좌표에 해당하는 지번 명칭
                if (resultCoordinate.bunji.length > 0) {
                  bunji = resultCoordinate.bunji;
                  newAddress += bunji + "\n";
                }
                // 새(도로명)주소 매칭을 한 경우, 길 이름을 반환
                if (resultCoordinate.newRoadName.length > 0) {
                  newRoadName = resultCoordinate.newRoadName;
                  newAddress += newRoadName + "\n";
                }
                // 새(도로명)주소 매칭을 한 경우, 건물 번호를 반환
                if (resultCoordinate.newBuildingIndex.length > 0) {
                  newBuildingIndex = resultCoordinate.newBuildingIndex;
                  newAddress += newBuildingIndex + "\n";
                }
                // 새(도로명)주소 매칭을 한 경우, 건물 이름를 반환
                if (resultCoordinate.newBuildingName.length > 0) {
                  newBuildingName = resultCoordinate.newBuildingName;
                  newAddress += newBuildingName + "\n";
                }
                // 새주소 건물을 매칭한 경우 새주소 건물 동을 반환
                if (resultCoordinate.newBuildingDong.length > 0) {
                  newBuildingDong = resultCoordinate.newBuildingDong;
                  newAddress += newBuildingDong + "\n";
                }
                // 검색 결과 표출
                if (lonEntr > 0) {
                  var docs = "<a style='color:orange' href='#webservice/docs/fullTextGeocoding'>Docs</a>"
                  var text = "검색결과(새주소) : " + newAddress + ",\n 응답코드:" + newMatchFlag + "(상세 코드 내역은 " + docs + " 에서 확인)" + "</br> 위경도좌표(중심점) : " + lat + ", " + lon + "</br>위경도좌표(입구점) : " + latEntr + ", " + lonEntr;
                  $("#result3").html(text);
                } else {
                  var docs = "<a style='color:orange' href='#webservice/docs/fullTextGeocoding'>Docs</a>"
                  var text = "검색결과(새주소) : " + newAddress + ",\n 응답코드:" + newMatchFlag + "(상세 코드 내역은 " + docs + " 에서 확인)" + "</br> 위경도좌표(입구점) : 위경도좌표(입구점)이 없습니다.";
                  $("#result3").html(text);
                }
              }

              // 구주소인 경우 newMatchFlag가 아닌 MatchFlag가 응닶값으로 온다
              if (resultCoordinate.matchFlag.length > 0) {
                // 매칭 구분 코드
                matchFlag = resultCoordinate.matchFlag;

                // 시/도 명칭
                if (resultCoordinate.city_do.length > 0) {
                  city = resultCoordinate.city_do;
                  address += city + "\n";
                }
                // 군/구 명칭
                if (resultCoordinate.gu_gun.length > 0) {
                  gu_gun = resultCoordinate.gu_gun;
                  address += gu_gun + "\n";
                }
                // 읍면동 명칭
                if (resultCoordinate.eup_myun.length > 0) {
                  eup_myun = resultCoordinate.eup_myun;
                  address += eup_myun + "\n";
                }
                // 출력 좌표에 해당하는 법정동 명칭
                if (resultCoordinate.legalDong.length > 0) {
                  legalDong = resultCoordinate.legalDong;
                  address += legalDong + "\n";
                }
                // 출력 좌표에 해당하는 행정동 명칭
                if (resultCoordinate.adminDong.length > 0) {
                  adminDong = resultCoordinate.adminDong;
                  address += adminDong + "\n";
                }
                // 출력 좌표에 해당하는 리 명칭
                if (resultCoordinate.ri.length > 0) {
                  ri = resultCoordinate.ri;
                  address += ri + "\n";
                }
                // 출력 좌표에 해당하는 지번 명칭
                if (resultCoordinate.bunji.length > 0) {
                  bunji = resultCoordinate.bunji;
                  address += bunji + "\n";
                }
                // 출력 좌표에 해당하는 건물 이름 명칭
                if (resultCoordinate.buildingName.length > 0) {
                  buildingName = resultCoordinate.buildingName;
                  address += buildingName + "\n";
                }
                // 출력 좌표에 해당하는 건물 동을 명칭
                if (resultCoordinate.buildingDong.length > 0) {
                  buildingDong = resultCoordinate.buildingDong;
                  address += buildingDong + "\n";
                }
                // 검색 결과 표출
                if (lonEntr > 0) {
                  var docs = "<a style='color:orange' href='#webservice/docs/fullTextGeocoding'>Docs</a>";
                  var text = "검색결과(지번주소) : " + address + "," + "\n" + "응답코드:" + matchFlag + "(상세 코드 내역은 " + docs + " 에서 확인)" + "</br>" + "위경도좌표(중심점) : " + lat + ", " + lon + "</br>" + "위경도좌표(입구점) : " + latEntr + ", " + lonEntr;
                  $("#result3").html(text);
                } else {
                  var docs = "<a style='color:orange' href='#webservice/docs/fullTextGeocoding'>Docs</a>";
                  var text = "검색결과(지번주소) : " + address + "," + "\n" + "응답코드:" + matchFlag + "(상세 코드 내역은 " + docs + " 에서 확인)" + "</br>" + "위경도좌표(입구점) : 위경도좌표(입구점)이 없습니다.";
                  $("#result3").html(text);
                }
              }
            }

          }
          , error: function (request, status, error) {
            console.log(request);
            console.log("code:" + request.status + "\n message:" + request.responseText + "\n error:" + error);
            $("#result3").html("error");
          }
        });

        //JSON TYPE EDIT [S&E]
        $
          .ajax({
            type: "POST",
            url: "https://apis.openapi.sk.com/tmap/routes?version=1&format=json&callback=result",
            async: false,
            data: {
              "appKey": "l7xxb7f655f2345248f7b5d6f198531b219a",
              "startX": loArray[0],
              "startY": laArray[0],
              "endX": loArray[1],
              "endY": laArray[1],
              "reqCoordType": "WGS84GEO",
              "resCoordType": "EPSG3857",
              "searchOption": searchOption,
              "trafficInfo": trafficInfochk
            },
            success: function (response) {

              //$("#result5").html(JSON.stringify(response));   //경로 JSON 데이터

              var resultData = response.features;

              var tDistance = "총 거리 : "
                + (resultData[0].properties.totalDistance / 1000)
                  .toFixed(1) + "km,";
              var tTime = " 총 시간 : "
                + (resultData[0].properties.totalTime / 60)
                  .toFixed(0) + "분,";
              var tFare = " 총 요금 : "
                + resultData[0].properties.totalFare
                + "원,";
              var taxiFare = " 예상 택시 요금 : "
                + resultData[0].properties.taxiFare
                + "원";

              $("#result").text(
                tDistance + tTime + tFare
                + taxiFare);
              ////////////////////////////////////////////////////////////////////////////////////////
              var RoadName = '';
              var num = 1;

              for (var i in resultData) {
                var geometry = resultData[i].geometry;
                var properties = resultData[i].properties;
                if (geometry.type == "LineString") {
                  //경로 도로명 나열
                  RoadName += num + " : " + properties.name + "\n";
                  num += 1;
                }
              }

              $("#result5").html(RoadName);


              ////////////////////////////////////////////////////////////////////////////////////////


              //교통정보 표출 옵션값을 체크
              if (trafficInfochk == "Y") {
                for (var i in resultData) { //for문 [S]
                  var geometry = resultData[i].geometry;
                  var properties = resultData[i].properties;

                  if (geometry.type == "LineString") {
                    //교통 정보도 담음
                    chktraffic
                      .push(geometry.traffic);
                    var sectionInfos = [];
                    var trafficArr = geometry.traffic;

                    for (var j in geometry.coordinates) {
                      // 경로들의 결과값들을 포인트 객체로 변환 
                      var latlng = new Tmapv2.Point(
                        geometry.coordinates[j][0],
                        geometry.coordinates[j][1]);
                      // 포인트 객체를 받아 좌표값으로 변환
                      var convertPoint = new Tmapv2.Projection.convertEPSG3857ToWGS84GEO(
                        latlng);

                      sectionInfos
                        .push(convertPoint);
                    }

                    drawLine(sectionInfos,
                      trafficArr);
                  } else {

                    var markerImg = "";
                    var pType = "";

                    if (properties.pointType == "S") { //출발지 마커
                      markerImg = "http://tmapapi.sktelecom.com/upload/tmap/marker/pin_r_m_s.png";
                      pType = "S";
                    } else if (properties.pointType == "E") { //도착지 마커
                      markerImg = "http://tmapapi.sktelecom.com/upload/tmap/marker/pin_r_m_e.png";
                      pType = "E";
                    } else { //각 포인트 마커
                      markerImg = "http://topopen.tmap.co.kr/imgs/point.png";
                      pType = "P"
                    }

                    // 경로들의 결과값들을 포인트 객체로 변환 
                    var latlon = new Tmapv2.Point(
                      geometry.coordinates[0],
                      geometry.coordinates[1]);
                    // 포인트 객체를 받아 좌표값으로 다시 변환
                    var convertPoint = new Tmapv2.Projection.convertEPSG3857ToWGS84GEO(
                      latlon);

                    var routeInfoObj = {
                      markerImage: markerImg,
                      lng: convertPoint._lng,
                      lat: convertPoint._lat,
                      pointType: pType
                    };
                    // 마커 추가
                    addMarkers(routeInfoObj);
                  }
                }//for문 [E]

              } else {

                for (var i in resultData) { //for문 [S]
                  var geometry = resultData[i].geometry;
                  var properties = resultData[i].properties;

                  if (geometry.type == "LineString") {
                    for (var j in geometry.coordinates) {
                      // 경로들의 결과값들을 포인트 객체로 변환 
                      var latlng = new Tmapv2.Point(
                        geometry.coordinates[j][0],
                        geometry.coordinates[j][1]);
                      // 포인트 객체를 받아 좌표값으로 변환
                      var convertPoint = new Tmapv2.Projection.convertEPSG3857ToWGS84GEO(
                        latlng);
                      // 포인트객체의 정보로 좌표값 변환 객체로 저장
                      var convertChange = new Tmapv2.LatLng(
                        convertPoint._lat,
                        convertPoint._lng);
                      // 배열에 담기
                      drawInfoArr
                        .push(convertChange);
                    }
                    drawLine(drawInfoArr,
                      "0");
                  } else {

                    var markerImg = "";
                    var pType = "";

                    if (properties.pointType == "S") { //출발지 마커
                      markerImg = "http://tmapapi.sktelecom.com/upload/tmap/marker/pin_r_m_s.png";
                      pType = "S";
                    } else if (properties.pointType == "E") { //도착지 마커
                      markerImg = "http://tmapapi.sktelecom.com/upload/tmap/marker/pin_r_m_e.png";
                      pType = "E";
                    } else { //각 포인트 마커
                      markerImg = "http://topopen.tmap.co.kr/imgs/point.png";
                      pType = "P"
                    }

                    // 경로들의 결과값들을 포인트 객체로 변환 
                    var latlon = new Tmapv2.Point(
                      geometry.coordinates[0],
                      geometry.coordinates[1]);
                    // 포인트 객체를 받아 좌표값으로 다시 변환
                    var convertPoint = new Tmapv2.Projection.convertEPSG3857ToWGS84GEO(
                      latlon);

                    var routeInfoObj = {
                      markerImage: markerImg,
                      lng: convertPoint._lng,
                      lat: convertPoint._lat,
                      pointType: pType
                    };

                    // Marker 추가
                    addMarkers(routeInfoObj);
                  }
                }//for문 [E]
              }
            },
            error: function (request, status, error) {
              console.log("code:"
                + request.status + "\n"
                + "message:"
                + request.responseText
                + "\n" + "error:" + error);
            }
          });
        //JSON TYPE EDIT [E]
      });
}

function addComma(num) {
  var regexp = /\B(?=(\d{3})+(?!\d))/g;
  return num.toString().replace(regexp, ',');
}

//마커 생성하기
function addMarkers(infoObj) {
  var size = new Tmapv2.Size(24, 38);//아이콘 크기 설정합니다.

  if (infoObj.pointType == "P") { //포인트점일때는 아이콘 크기를 줄입니다.
    size = new Tmapv2.Size(8, 8);
  }

  marker_p = new Tmapv2.Marker({
    position: new Tmapv2.LatLng(infoObj.lat, infoObj.lng),
    icon: infoObj.markerImage,
    iconSize: size,
    map: map
  });

  resultMarkerArr.push(marker_p);
}

//라인그리기
function drawLine(arrPoint, traffic) {
  var polyline_;

  if (chktraffic.length != 0) {

    // 교통정보 혼잡도를 체크
    // strokeColor는 교통 정보상황에 다라서 변화
    // traffic :  0-정보없음, 1-원활, 2-서행, 3-지체, 4-정체  (black, green, yellow, orange, red)

    var lineColor = "";

    if (traffic != "0") {
      if (traffic.length == 0) { //length가 0인것은 교통정보가 없으므로 검은색으로 표시

        lineColor = "#06050D";
        //라인그리기[S]
        polyline_ = new Tmapv2.Polyline({
          path: arrPoint,
          strokeColor: lineColor,
          strokeWeight: 6,
          map: map
        });
        resultdrawArr.push(polyline_);
        //라인그리기[E]
      } else { //교통정보가 있음

        if (traffic[0][0] != 0) { //교통정보 시작인덱스가 0이 아닌경우
          var trafficObject = "";
          var tInfo = [];

          for (var z = 0; z < traffic.length; z++) {
            trafficObject = {
              "startIndex": traffic[z][0],
              "endIndex": traffic[z][1],
              "trafficIndex": traffic[z][2],
            };
            tInfo.push(trafficObject)
          }

          var noInfomationPoint = [];

          for (var p = 0; p < tInfo[0].startIndex; p++) {
            noInfomationPoint.push(arrPoint[p]);
          }

          //라인그리기[S]
          polyline_ = new Tmapv2.Polyline({
            path: noInfomationPoint,
            strokeColor: "#06050D",
            strokeWeight: 6,
            map: map
          });
          //라인그리기[E]
          resultdrawArr.push(polyline_);

          for (var x = 0; x < tInfo.length; x++) {
            var sectionPoint = []; //구간선언

            for (var y = tInfo[x].startIndex; y <= tInfo[x].endIndex; y++) {
              sectionPoint.push(arrPoint[y]);
            }

            if (tInfo[x].trafficIndex == 0) {
              lineColor = "#06050D";
            } else if (tInfo[x].trafficIndex == 1) {
              lineColor = "#61AB25";
            } else if (tInfo[x].trafficIndex == 2) {
              lineColor = "#FFFF00";
            } else if (tInfo[x].trafficIndex == 3) {
              lineColor = "#E87506";
            } else if (tInfo[x].trafficIndex == 4) {
              lineColor = "#D61125";
            }

            //라인그리기[S]
            polyline_ = new Tmapv2.Polyline({
              path: sectionPoint,
              strokeColor: lineColor,
              strokeWeight: 6,
              map: map
            });
            //라인그리기[E]
            resultdrawArr.push(polyline_);
          }
        } else { //0부터 시작하는 경우

          var trafficObject = "";
          var tInfo = [];

          for (var z = 0; z < traffic.length; z++) {
            trafficObject = {
              "startIndex": traffic[z][0],
              "endIndex": traffic[z][1],
              "trafficIndex": traffic[z][2],
            };
            tInfo.push(trafficObject)
          }

          for (var x = 0; x < tInfo.length; x++) {
            var sectionPoint = []; //구간선언

            for (var y = tInfo[x].startIndex; y <= tInfo[x].endIndex; y++) {
              sectionPoint.push(arrPoint[y]);
            }

            if (tInfo[x].trafficIndex == 0) {
              lineColor = "#06050D";
            } else if (tInfo[x].trafficIndex == 1) {
              lineColor = "#61AB25";
            } else if (tInfo[x].trafficIndex == 2) {
              lineColor = "#FFFF00";
            } else if (tInfo[x].trafficIndex == 3) {
              lineColor = "#E87506";
            } else if (tInfo[x].trafficIndex == 4) {
              lineColor = "#D61125";
            }

            //라인그리기[S]
            polyline_ = new Tmapv2.Polyline({
              path: sectionPoint,
              strokeColor: lineColor,
              strokeWeight: 6,
              map: map
            });
            //라인그리기[E]
            resultdrawArr.push(polyline_);
          }
        }
      }
    } else {

    }
  } else {
    polyline_ = new Tmapv2.Polyline({
      path: arrPoint,
      strokeColor: "#DD0000",
      strokeWeight: 6,
      map: map
    });
    resultdrawArr.push(polyline_);
  }

}

//초기화 기능
function resettingMap() {
  //기존마커는 삭제
  marker_s.setMap(null);
  marker_e.setMap(null);
  if (resultMarkerArr.length > 0) {
    for (var i = 0; i < resultMarkerArr.length; i++) {
      resultMarkerArr[i].setMap(null);
    }
  }

  if (resultdrawArr.length > 0) {
    for (var i = 0; i < resultdrawArr.length; i++) {
      resultdrawArr[i].setMap(null);
    }
  }

  chktraffic = [];
  drawInfoArr = [];
  resultMarkerArr = [];
  resultdrawArr = [];
  loArray = [];
  laArray = [];
}