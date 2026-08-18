<launch>
  <!-- Запуск симуляции и навигации -->
  
  <!-- Узел симуляции -->
  <node pkg="aerial_nav_ros" exec="sim_node" name="aerial_simulation" output="screen">
    <param name="map_path" value="$(env HOME)/aerial-nav/map_cache/real_map.png"/>
    <param name="resolution" value="2.35"/>
    <param name="drone_speed" value="5.0"/>
    <param name="drone_height" value="50.0"/>
  </node>

  <!-- Узел навигации -->
  <node pkg="aerial_nav_ros" exec="nav_node" name="aerial_navigation" output="screen">
    <param name="map_path" value="$(env HOME)/aerial-nav/map_cache/real_map.png"/>
    <param name="resolution" value="2.35"/>
    <param name="use_cuda" value="false"/>
  </node>

  <!-- RViz для визуализации -->
  <node pkg="rviz2" exec="rviz2" name="rviz2" args="-d $(find-pkg-share aerial_nav_ros)/rviz/nav.rviz"/>
</launch>
