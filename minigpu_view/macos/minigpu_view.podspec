Pod::Spec.new do |s|
  s.name             = 'minigpu_view'
  s.version          = '0.0.1'
  s.summary          = 'Zero-copy GPU preview widget for miniav/minigpu (macOS stub).'
  s.description      = <<-DESC
Stub macOS implementation; all method calls return PlatformException(code: "unsupported").
                       DESC
  s.homepage         = 'https://example.com'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'miniav' => 'noreply@example.com' }

  s.source           = { :path => '.' }
  s.source_files     = 'Classes/**/*'
  s.dependency 'FlutterMacOS'

  s.platform = :osx, '10.14'
  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES' }
  s.swift_version = '5.0'
end
